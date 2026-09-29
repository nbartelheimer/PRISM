#include "prism.h"

#include <assert.h>
#include <limits.h>
#include <string.h>
#include <mpi.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <stdio.h>
#include <stdlib.h>

#define PRISM_ERROR -1
#define PRISM_SUCCESS 0

//#define DEBUG
#ifdef DEBUG
#define OPT_PERSISTENT_INFO(fmt, ...)                                  \
	fprintf(stdout, "[OPT PERSISTENT INFO] %s:%d:%s(): " fmt "\n", \
		__FILE__, __LINE__, __func__, ##__VA_ARGS__);          \
	fflush(stdout)
#else
#define OPT_PERSISTENT_INFO(fmt, ...)
#endif

#define OPT_PERSISTENT_ERR(fmt, ...)                                    \
	fprintf(stderr, "[OPT PERSISTENT ERROR] %s:%d:%s(): " fmt "\n", \
		__FILE__, __LINE__, __func__, ##__VA_ARGS__)

#define READY_TO_RECEIVE_FLAG 98
#define TRANSMISSION_COMPLETE_FLAG 99

#define PERSISTENT_LIKELY(x) __builtin_expect(!!(x), 1)
#define PERSISTENT_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define MPIEXT_PERSISTENT_ADDR_KEY "OPT_PERSISTENT_ADDR"
#define PERSISTENT_ALWAYS_INLINE __attribute__((always_inline))

#define MPIEXT_PERSISTENT_DEFAULT_COMP 64
#define DEFAULT_REQUEST_STORE_SIZE UINT32_C(512)

typedef enum mpiext_persistent_request_init_state {
	CHECK_FOR_REMOTE_INFO = 1,
	READY
} mpiext_persistent_request_init_state_t;

typedef enum mpiext_persistent_op_type {
	PERSISTENT_SEND = 1,
	PERSISTENT_RECV,
	OP_INVALID
} mpiext_persistent_op_type_t;

typedef enum request_store_state {
	REQ_ACTIVE = 1,
	REQ_FREE
} request_store_state_t;

typedef struct mpiext_persistent_ctx {
	struct fi_info *fi;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_cq *cq;
	struct fid_av *av;
	struct fid_ep *ep;
	fi_addr_t *peer_addr;
	int n_peers;
	int my_world_rank;
} mpiext_persistent_ctx_t;

typedef struct persistent_request {
	/* fi_context2 also provides the storage required by FI_CONTEXT. */
	struct fi_context2 context;
	mpiext_persistent_ctx_t *transport;
	fi_addr_t peer_addr;
	uint64_t remote_data_addr;
	uint64_t remote_flag_addr;
	uint64_t remote_data_mkey;
	uint64_t remote_flag_mkey;
	uint64_t posted_ops;
	uint64_t completed_ops;
	size_t size;
	size_t my_rdma_info_size;
	struct fid_mr *data_buffer_mr;
	struct fid_mr *flag_buffer_mr;
	void *data_buffer;
	void *my_rdma_info_buffer;
	uint32_t remote_store_index;
	int flag_buffer;
	int tag;
	int peer_rank;
	int store_index;
	MPI_Comm req_comm;
	mpiext_persistent_request_init_state_t init_state;
	mpiext_persistent_op_type_t op_type;
	request_store_state_t req_state;
} persistent_request_t;

typedef struct request_element {
	uint32_t index;
	struct request_element *next;
} request_element_t;

static mpiext_persistent_ctx_t ctx;
static mpiext_persistent_ctx_t shm_ctx;
static unsigned char *local_peers;
static int verbose;

static persistent_request_t *request_store = NULL;
static request_element_t *free_requests_head = NULL;
static request_element_t *free_requests_tail = NULL;
static request_element_t *request_element_mpool = NULL;

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_cleanup_request_ressources(persistent_request_t *req) {
	if (req->data_buffer_mr) fi_close(&req->data_buffer_mr->fid);
	if (req->flag_buffer_mr) fi_close(&req->flag_buffer_mr->fid);
	free(req->my_rdma_info_buffer);
}

static inline void free_requests_append(request_element_t *elem) {
	if (free_requests_head == NULL) {
		free_requests_head = elem;
		free_requests_tail = elem;

	} else {
		free_requests_tail->next = elem;
		free_requests_tail = elem;
	}
}

// forward declaration
static inline void mpiext_persistent_reset_request(
    persistent_request_t *request);

static inline int init_request_storage(void) {
	request_element_t *elem = NULL;

	request_store =
	    calloc(DEFAULT_REQUEST_STORE_SIZE, sizeof(persistent_request_t));
	if (!request_store) {
		OPT_PERSISTENT_ERR("request_store alloc failed");
		return -1;
	}

	for (uint32_t i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
		request_store[i].req_state = REQ_FREE;
		request_store[i].store_index = i;
		mpiext_persistent_reset_request(&request_store[i]);

		elem = malloc(sizeof(request_element_t));

		if (!elem) {
			OPT_PERSISTENT_ERR("elem alloc failed");
			return -1;
		}

		elem->index = i;
		elem->next = NULL;

		free_requests_append(elem);
	}

	return PRISM_SUCCESS;
}

static inline void cleanup_request_storage(void) {
	request_element_t *elem = free_requests_head;

	while (elem != NULL) {
		request_element_t *next = elem->next;
		free(elem);
		elem = next;
	}

	free_requests_head = NULL;
	free_requests_tail = NULL;

	elem = request_element_mpool;

	while (elem != NULL) {
		request_element_t *next = elem->next;
		free(elem);
		elem = next;
	}

	request_element_mpool = NULL;

	for (uint32_t i = 0; request_store && i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
		if (request_store[i].req_state != REQ_ACTIVE) continue;

		mpiext_persistent_cleanup_request_ressources(&request_store[i]);
	}

	free(request_store);
	request_store = NULL;
}

static inline void request_element_mpool_put(request_element_t *elem) {
	elem->index = -1;
	elem->next = NULL;

	if (request_element_mpool == NULL) {
		request_element_mpool = elem;
	} else {
		// LIFO
		elem->next = request_element_mpool;
		request_element_mpool = elem;
	}
}

static inline request_element_t *request_element_mpool_get(void) {
	request_element_t *elem = request_element_mpool;
	request_element_mpool = elem->next;
	return elem;
}

static inline persistent_request_t *get_persistent_request(void) {
	request_element_t *elem = free_requests_head;
	if (!elem) return NULL;
	free_requests_head = elem->next;
	if (!free_requests_head) free_requests_tail = NULL;
	elem->next = NULL;
	persistent_request_t *req = &request_store[elem->index];

	req->req_state = REQ_ACTIVE;
	req->store_index = elem->index;

	request_element_mpool_put(elem);

	return req;
}

static inline void put_persistent_request(int index) {
	request_element_t *elem = request_element_mpool_get();

	elem->index = index;

	mpiext_persistent_cleanup_request_ressources(&request_store[index]);
	mpiext_persistent_reset_request(&request_store[index]);

	request_store[index].req_state = REQ_FREE;

	free_requests_append(elem);
}

#ifdef DEBUG
static void mpiext_persistent_print_provider_info(struct fi_info *info) {
	if (NULL != info) {
		fprintf(stderr, "Provider info:\n");
		fprintf(stderr, "   %s (%s)\n", info->fabric_attr->prov_name,
			info->domain_attr->name);

		fprintf(stderr, "  capabilities: %s\n",
			fi_tostr(&info->caps, FI_TYPE_CAPS));
		fprintf(stderr, "  mode: %s\n",
			fi_tostr(&info->mode, FI_TYPE_MODE));
		if (info->mode & FI_CONTEXT) fprintf(stderr, "FI_CONTEXT\n");
		if (info->mode & FI_CONTEXT2) fprintf(stderr, "FI_CONTEXT2\n");

		struct fi_domain_attr *domain_attr = info->domain_attr;
		fprintf(stderr, "Domain attributes:\n");
		fprintf(stderr, "  threading: %s\n",
			fi_tostr(&domain_attr->threading, FI_TYPE_THREADING));
		fprintf(
		    stderr, "  data progress mode %s\n",
		    fi_tostr(&domain_attr->data_progress, FI_TYPE_PROGRESS));
		fprintf(
		    stderr, "  control progress mode %s\n",
		    fi_tostr(&domain_attr->control_progress, FI_TYPE_PROGRESS));
		fprintf(stderr, "  MR mode %s\n",
			fi_tostr(&domain_attr->mr_mode, FI_TYPE_MR_MODE));

		struct fi_tx_attr *tx_attr = info->tx_attr;
		fprintf(stderr, "Tx attributes:\n");
		fprintf(stderr, "  tx iov_limit: %ld\n", tx_attr->iov_limit);
		fprintf(stderr, "  tx rma_iov_limit: %ld\n",
			tx_attr->rma_iov_limit);
	}
}
#endif

static inline void mpiext_persistent_reset_request(
    persistent_request_t *request) {
	memset(&request->context, 0, sizeof(request->context));
	request->transport = NULL;
	request->peer_addr = FI_ADDR_UNSPEC;
	request->posted_ops = 0;
	request->completed_ops = 0;
	request->remote_data_addr = 0;
	request->remote_flag_addr = 0;
	request->remote_data_mkey = 0;
	request->remote_flag_mkey = 0;
	request->data_buffer_mr = NULL;
	request->flag_buffer_mr = NULL;
	request->size = 0;
	request->my_rdma_info_size = 0;
	request->data_buffer = NULL;
	request->my_rdma_info_buffer = NULL;
	request->flag_buffer = -1;
	request->tag = -1;
	request->req_comm = MPI_COMM_NULL;
	request->peer_rank = -1;
	request->init_state = CHECK_FOR_REMOTE_INFO;
	request->op_type = OP_INVALID;
}

static void mpiext_persistent_cleanup_module(void);

/* Keep collective setup failures consistent so a peer cannot enter an
 * address exchange after another peer has already left initialization. */
static int setup_status(MPI_Comm comm, int ret) {
	int failed = ret != 0, any_failed;
	PMPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
	return any_failed ? (ret ? ret : PRISM_ERROR) : PRISM_SUCCESS;
}

static void close_transport(mpiext_persistent_ctx_t *transport) {
	if (transport->ep) fi_close(&transport->ep->fid);
	if (transport->av) fi_close(&transport->av->fid);
	if (transport->cq) fi_close(&transport->cq->fid);
	if (transport->domain) fi_close(&transport->domain->fid);
	if (transport->fabric) fi_close(&transport->fabric->fid);
	if (transport->fi) fi_freeinfo(transport->fi);
	free(transport->peer_addr);
	*transport = (mpiext_persistent_ctx_t){0};
}

static int open_transport(mpiext_persistent_ctx_t *transport, int shared,
                          int world_size) {
	int ret;
	struct fi_info *hints = fi_allocinfo();
	struct fi_cq_attr cq_attr = {0};
	struct fi_av_attr av_attr = {0};
	if (!hints) return -FI_ENOMEM;

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_LOCAL_COMM |
	              (shared ? 0 : FI_REMOTE_COMM);
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT |
	                              FI_MR_ALLOCATED | FI_MR_PROV_KEY |
	                              FI_MR_VIRT_ADDR;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	/* A request index is carried in every remote data completion. */
	hints->domain_attr->cq_data_size = sizeof(uint32_t);
	hints->tx_attr->inject_size = sizeof(int);
	if (shared) {
		hints->fabric_attr->prov_name = strdup("shm");
		if (!hints->fabric_attr->prov_name) {
			fi_freeinfo(hints);
			return -FI_ENOMEM;
		}
	}
	ret = fi_getinfo(FI_VERSION(2, 6), NULL, NULL, 0, hints, &transport->fi);
	fi_freeinfo(hints);
	if (ret) return ret;

	ret = fi_fabric(transport->fi->fabric_attr, &transport->fabric, NULL);
	if (ret) return ret;
	ret = fi_domain(transport->fabric, transport->fi, &transport->domain, NULL);
	if (ret) return ret;
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.wait_obj = FI_WAIT_NONE;
	ret = fi_cq_open(transport->domain, &cq_attr, &transport->cq, NULL);
	if (ret) return ret;
	av_attr.type = FI_AV_TABLE;
	av_attr.count = world_size;
	ret = fi_av_open(transport->domain, &av_attr, &transport->av, NULL);
	if (ret) return ret;
	ret = fi_endpoint(transport->domain, transport->fi, &transport->ep, NULL);
	if (ret) return ret;
	ret = fi_ep_bind(transport->ep, &transport->cq->fid,
	                 FI_TRANSMIT | FI_SELECTIVE_COMPLETION | FI_RECV);
	if (ret) return ret;
	ret = fi_ep_bind(transport->ep, &transport->av->fid, 0);
	if (ret) return ret;
	return fi_enable(transport->ep);
}

/* shm names are variable-length strings. Exchange their actual lengths,
 * and restrict their AV to ranks in the MPI shared-memory communicator. */
static int exchange_addresses(mpiext_persistent_ctx_t *transport,
                              MPI_Comm comm, int world_size) {
	int ret, count, rank, total = 0, length = 0;
	size_t addr_len = 0;
	char *addr = NULL, *addresses = NULL;
	int *lengths = NULL, *offsets = NULL, *ranks = NULL;
	PMPI_Comm_size(comm, &count);
	PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
	ret = fi_getname(&transport->ep->fid, NULL, &addr_len);
	if (ret == -FI_ETOOSMALL && addr_len > 0 && addr_len <= INT_MAX) {
		addr = malloc(addr_len);
		ret = addr ? fi_getname(&transport->ep->fid, addr, &addr_len) : -FI_ENOMEM;
	} else if (!ret) {
		ret = -FI_EINVAL;
	}
	lengths = calloc(count, sizeof(int));
	offsets = calloc(count, sizeof(int));
	ranks = calloc(count, sizeof(int));
	transport->peer_addr = malloc(world_size * sizeof(fi_addr_t));
	if (!lengths || !offsets || !ranks || !transport->peer_addr) ret = -FI_ENOMEM;
	ret = setup_status(comm, ret);
	if (ret) goto out;
	for (int i = 0; i < world_size; ++i) transport->peer_addr[i] = FI_ADDR_UNSPEC;
	length = (int)addr_len;
	PMPI_Allgather(&length, 1, MPI_INT, lengths, 1, MPI_INT, comm);
	PMPI_Allgather(&rank, 1, MPI_INT, ranks, 1, MPI_INT, comm);
	for (int i = 0; i < count; ++i) {
		if (lengths[i] <= 0 || lengths[i] > INT_MAX - total) {
			ret = -FI_EINVAL;
			break;
		}
		offsets[i] = total;
		total += lengths[i];
	}
	if (!ret) {
		addresses = malloc(total);
		if (!addresses) ret = -FI_ENOMEM;
	}
	ret = setup_status(comm, ret);
	if (ret) goto out;
	PMPI_Allgatherv(addr, length, MPI_BYTE, addresses, lengths, offsets, MPI_BYTE, comm);
	for (int i = 0; i < count; ++i) {
		ret = fi_av_insert(transport->av, addresses + offsets[i], 1,
		                   &transport->peer_addr[ranks[i]], 0, NULL);
		if (ret != 1) {
			if (ret >= 0) ret = -FI_EIO;
			break;
		}
		ret = 0;
	}
	ret = setup_status(comm, ret);
out:
	free(addr);
	free(addresses);
	free(lengths);
	free(offsets);
	free(ranks);
	return ret;
}

int PRISM_Init(void) {
	int ret, my_rank, world_size, local_size;
	int *ranks = NULL;
	MPI_Comm local_comm = MPI_COMM_NULL;
	verbose = getenv("PRISM_VERBOSE") && strcmp(getenv("PRISM_VERBOSE"), "0");
	PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	PMPI_Comm_size(MPI_COMM_WORLD, &world_size);
	PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank,
	                     MPI_INFO_NULL, &local_comm);
	PMPI_Comm_size(local_comm, &local_size);
	local_peers = calloc(world_size, sizeof(*local_peers));
	ranks = malloc(local_size * sizeof(*ranks));
	ret = setup_status(MPI_COMM_WORLD, local_peers && ranks ? 0 : -FI_ENOMEM);
	if (ret) goto fail;
	PMPI_Allgather(&my_rank, 1, MPI_INT, ranks, 1, MPI_INT, local_comm);
	for (int i = 0; i < local_size; ++i) local_peers[ranks[i]] = 1;
	free(ranks);
	ranks = NULL;
	/* shm is required and initialized on every rank, including singleton nodes. */
	ret = open_transport(&shm_ctx, 1, local_size);
	if (ret) goto fail;
	ret = exchange_addresses(&shm_ctx, local_comm, world_size);
	if (ret) goto fail;
	/* This condition is identical on all ranks: the job spans multiple nodes. */
	if (local_size != world_size) {
		ret = open_transport(&ctx, 0, world_size);
		ret = setup_status(MPI_COMM_WORLD, ret);
		if (!ret) ret = exchange_addresses(&ctx, MPI_COMM_WORLD, world_size);
		if (ret) goto fail;
	}
	ctx.my_world_rank = my_rank;
	ctx.n_peers = world_size;
#ifdef DEBUG
	if (my_rank == 0) {
		mpiext_persistent_print_provider_info(ctx.fi);
		mpiext_persistent_print_provider_info(shm_ctx.fi);
	}
#endif
	ret = setup_status(MPI_COMM_WORLD, init_request_storage());
	if (ret) {
		cleanup_request_storage();
		goto fail;
	}
	if (verbose) {
		fprintf(stderr, "PRISM rank %d: network=%s, local=shm (%d local ranks)\n",
		        my_rank, ctx.fi ? ctx.fi->fabric_attr->prov_name : "unused",
		        local_size);
	}
	PMPI_Comm_free(&local_comm);
	return PRISM_SUCCESS;
fail:
	OPT_PERSISTENT_ERR("PRISM_Init failed: %s (%d)", fi_strerror(-ret), ret);
	free(ranks);
	if (local_comm != MPI_COMM_NULL) PMPI_Comm_free(&local_comm);
	mpiext_persistent_cleanup_module();
	return ret;
}

static int register_buffer(mpiext_persistent_ctx_t *transport, void *buffer,
                           size_t size, uint64_t key, struct fid_mr **mr) {
	int ret = fi_mr_reg(transport->domain, buffer, size,
	                    FI_WRITE | FI_REMOTE_WRITE, 0,
	                    (transport->fi->domain_attr->mr_mode & FI_MR_PROV_KEY) ? 0 : key,
	                    0, mr, NULL);
	if (!ret && (transport->fi->domain_attr->mr_mode & FI_MR_ENDPOINT)) {
		ret = fi_mr_bind(*mr, &transport->ep->fid, 0);
		if (!ret) ret = fi_mr_enable(*mr);
	}
	if (ret) OPT_PERSISTENT_ERR("MR registration failed: %s", fi_strerror(-ret));
	return ret;
}

static inline int mpiext_persistent_register_memory(persistent_request_t *request) {
	/* shm uses application keys: data and flag MRs must have distinct keys,
	 * including when multiple requests register the same buffer. */
	uint64_t key = 2 * (uint64_t)request->store_index + 1;
	int ret = register_buffer(request->transport,
	                          request->size ? request->data_buffer : &request->flag_buffer,
	                          request->size ? request->size : sizeof(int),
	                          key, &request->data_buffer_mr);
	if (ret) return ret;
	return register_buffer(request->transport, &request->flag_buffer, sizeof(int),
	                       key + 1, &request->flag_buffer_mr);
}

PERSISTENT_ALWAYS_INLINE static inline void mpiext_persistent_pack_rdma_info(
    persistent_request_t *request) {
	uint64_t *packed = request->my_rdma_info_buffer;
	int virtual_addr = request->transport->fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR;
	packed[0] = virtual_addr ? (uintptr_t)(request->size ? request->data_buffer : &request->flag_buffer) : 0;
	packed[1] = fi_mr_key(request->data_buffer_mr);
	packed[2] = virtual_addr ? (uintptr_t)&request->flag_buffer : 0;
	packed[3] = fi_mr_key(request->flag_buffer_mr);
	packed[4] = (uint64_t)request->store_index;
}

PERSISTENT_ALWAYS_INLINE static inline void mpiext_persistent_unpack_rdma_info(
    persistent_request_t *request, void *payload) {
	uint64_t *packed = payload;
	request->remote_data_addr = packed[0];
	request->remote_data_mkey = packed[1];
	request->remote_flag_addr = packed[2];
	request->remote_flag_mkey = packed[3];
	request->remote_store_index = (uint32_t)packed[4];
}

static void mpiext_persistent_cleanup_module(void) {
	close_transport(&shm_ctx);
	close_transport(&ctx);
	free(local_peers);
	local_peers = NULL;
}

int PRISM_Finalize(void) {
	cleanup_request_storage();
	mpiext_persistent_cleanup_module();
	return PRISM_SUCCESS;
}

static int mpiext_persistent_init_request(persistent_request_t *request) {
	int world_rank, ret;
	MPI_Group group, world_group;
	/* API peers are communicator ranks; AV indices are world ranks. */
	PMPI_Comm_group(request->req_comm, &group);
	PMPI_Comm_group(MPI_COMM_WORLD, &world_group);
	PMPI_Group_translate_ranks(group, 1, &request->peer_rank, world_group, &world_rank);
	PMPI_Group_free(&group);
	PMPI_Group_free(&world_group);
	if (world_rank == MPI_UNDEFINED || world_rank < 0 || world_rank >= ctx.n_peers)
		return PRISM_ERROR;
	request->transport = local_peers[world_rank] ? &shm_ctx : &ctx;
	request->peer_addr = request->transport->peer_addr[world_rank];
	ret = mpiext_persistent_register_memory(request);
	if (ret) return ret;
	request->my_rdma_info_size = 5 * sizeof(uint64_t);

	request->my_rdma_info_buffer = malloc(request->my_rdma_info_size);
	if (!request->my_rdma_info_buffer) return -FI_ENOMEM;

	mpiext_persistent_pack_rdma_info(request);

	// We assume eager protocol usage here
	PMPI_Send(request->my_rdma_info_buffer, request->my_rdma_info_size,
		  MPI_BYTE, request->peer_rank, request->tag, request->req_comm);

	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int mpiext_persistent_progress_transport(
    mpiext_persistent_ctx_t *transport, int count, struct fi_cq_data_entry cqes[]) {
	struct fi_cq_err_entry err = {0};
	persistent_request_t *request = NULL;
	ssize_t rc;

	if (!transport->cq) return 0;
	rc = fi_cq_read(transport->cq, cqes, count);

	if (rc > 0) {
		for (ssize_t i = 0; i < rc; ++i) {
			if (cqes[i].flags & FI_REMOTE_WRITE) {
				OPT_PERSISTENT_INFO(
				    "Received message with index: %u",
				    (uint32_t)cqes[i].data);
				request =
				    &request_store[(uint32_t)cqes[i].data];
				request->completed_ops++;
			} else if (cqes[i].flags & FI_WRITE) {
				request = cqes[i].op_context;
				request->completed_ops++;
			}
		}
	}

	if (PERSISTENT_UNLIKELY(rc == -FI_EAVAIL)) {
		fi_cq_readerr(transport->cq, &err, 0);
		OPT_PERSISTENT_ERR("CQ: %s",
				   fi_cq_strerror(transport->cq, err.prov_errno,
						  err.err_data, NULL, 0));
		return rc;
	}

	return rc < 0 && rc != -FI_EAGAIN ? (int)rc : 0;
}

PERSISTENT_ALWAYS_INLINE static inline int mpiext_persistent_progress(
    int count, struct fi_cq_data_entry cqes[]) {
	int ret = mpiext_persistent_progress_transport(&ctx, count, cqes);
	int shm_ret = mpiext_persistent_progress_transport(&shm_ctx, count, cqes);
	return ret ? ret : shm_ret;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_data_transfer(persistent_request_t *request) {
	ssize_t ret = -1;
	void *desc = fi_mr_desc(request->data_buffer_mr);
	uint64_t flags = 0;
	struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];

	struct fi_rma_iov rma_iov = {
	    .addr = request->remote_data_addr,
	    .len = request->size,
	    .key = request->remote_data_mkey,
	};

	struct iovec iov = {
	    .iov_base = request->data_buffer,
	    .iov_len = request->size,
	};

	struct fi_msg_rma rma_msg = {
	    .msg_iov = &iov,
	    .desc = &desc,
	    .iov_count = 1,
	    .addr = request->peer_addr,
	    .rma_iov = &rma_iov,
	    .rma_iov_count = 1,
	    .context = (void *)request,
	    .data = request->remote_store_index,
	};

	flags = FI_REMOTE_CQ_DATA;

	if (request->size <= request->transport->fi->tx_attr->inject_size) {
		flags |= FI_INJECT;
	} else {
		flags |= FI_COMPLETION;
		request->posted_ops++;
	}

	do {
		ret = fi_writemsg(request->transport->ep, &rma_msg, flags);
		if (ret == -FI_EAGAIN) {
			int progress_ret = mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
			if (progress_ret) return progress_ret;
		} else if (ret < 0) {
			OPT_PERSISTENT_ERR("fi_writemsg data failed with %s",
					   fi_strerror(-ret));
			return ret;
		}
	} while (ret);

	return 0;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_flag_transfer(persistent_request_t *request) {
	struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];
	ssize_t ret = -1;

	do {
		ret = fi_inject_write(
		    request->transport->ep, &request->flag_buffer, sizeof(int),
		    request->peer_addr,
		    request->remote_flag_addr, request->remote_flag_mkey);
		if (ret == -FI_EAGAIN) {
			int progress_ret = mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
			if (progress_ret) return progress_ret;
		} else if (ret < 0) {
			OPT_PERSISTENT_ERR("fi_writemsg data failed with %s",
					   fi_strerror(-ret));
			return ret;
		}
	} while (ret);

	return 0;
}

static inline int mpiext_persistent_start_send_core_no_sync(
    persistent_request_t *request) {
	return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_start_send_core(
    persistent_request_t *request) {
	struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];
	volatile int *flag = &request->flag_buffer;
	OPT_PERSISTENT_INFO("Waiting for RTR flag on %i from %i on addr %p\n",
			    ctx.my_world_rank, request->peer_rank, &request->flag_buffer);
	while (*flag != READY_TO_RECEIVE_FLAG) {
		int progress_ret = mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
		if (progress_ret) return progress_ret;
	}
	request->flag_buffer = 0;
	OPT_PERSISTENT_INFO("Got RTR flag\n");

	return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_finalize_send_core(
    persistent_request_t *request) {
	struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];

	while (request->posted_ops != request->completed_ops) {
		int ret = mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
		if (ret) return ret;
	}

	OPT_PERSISTENT_INFO("SEND OPERATION FINALIZED");
	return PRISM_SUCCESS;
}

static inline int mpiext_persistent_start_recv_core_no_sync(
    persistent_request_t *request) {
	request->posted_ops++;
	return 0;
}

static inline int mpiext_persistent_start_recv_core(
    persistent_request_t *request) {
	request->flag_buffer = READY_TO_RECEIVE_FLAG;
	request->posted_ops++;
	OPT_PERSISTENT_INFO("Sending RTR flag to %i from %i to addr 0x%lx\n",
			    request->peer_rank, ctx.my_world_rank, request->remote_flag_addr);
	return mpiext_persistent_start_flag_transfer(request);
}

static inline int mpiext_persistent_finalize_recv_core(
    persistent_request_t *request) {
	struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];
	OPT_PERSISTENT_INFO("Waiting for DATA");
	while (request->posted_ops != request->completed_ops) {
		int ret = mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
		if (ret) return ret;
	}
	OPT_PERSISTENT_INFO("Finalized Recv operation");
	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first_no_sync(persistent_request_t *request) {
	void *payload = malloc(request->my_rdma_info_size);
	assert(payload != NULL);

	PMPI_Recv(payload, request->my_rdma_info_size, MPI_BYTE,
		  request->peer_rank, request->tag, request->req_comm,
		  MPI_STATUS_IGNORE);
	mpiext_persistent_unpack_rdma_info(request, payload);
	free(payload);

	// important: set ready state here
	request->init_state = READY;

	// Start operation after we finished init
	if (request->op_type == PERSISTENT_SEND) {
		OPT_PERSISTENT_INFO("ready send operation on %i",
				    ctx.my_world_rank);
		return mpiext_persistent_start_send_core_no_sync(request);
	} else {
		OPT_PERSISTENT_INFO("ready recv operation on %i\n",
				    ctx.my_world_rank);
		return mpiext_persistent_start_recv_core_no_sync(request);
	}
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first(persistent_request_t *request) {
	void *payload = malloc(request->my_rdma_info_size);
	assert(payload != NULL);

	PMPI_Recv(payload, request->my_rdma_info_size, MPI_BYTE,
		  request->peer_rank, request->tag, request->req_comm,
		  MPI_STATUS_IGNORE);
	mpiext_persistent_unpack_rdma_info(request, payload);

	free(payload);

	// important: check ready state here
	request->init_state = READY;

	// Start operation after we finished init
	if (request->op_type == PERSISTENT_SEND) {
		OPT_PERSISTENT_INFO("ready send operation on %i",
				    ctx.my_world_rank);
		return mpiext_persistent_start_send_core(request);
	} else {
		OPT_PERSISTENT_INFO("ready recv operation on %i\n",
				    ctx.my_world_rank);
		return mpiext_persistent_start_recv_core(request);
	}
}

int PRISM_Psend_init(const void *buffer, int count, MPI_Datatype datatype,
		     int dst, int tag, MPI_Comm communicator,
		     PRISM_Request *request) {
	int size;

	assert(buffer != NULL);

	persistent_request_t *req = get_persistent_request();

	if (!req) return -FI_ENOMEM;

	req->req_comm = communicator;
	req->tag = tag;
	req->peer_rank = dst;
	req->data_buffer = (void *)buffer;
	req->op_type = PERSISTENT_SEND;

	PMPI_Type_size(datatype, &size);
	req->size = (size_t)size * count;

	*request = (void *)req;
	OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p", (void *)buffer);
	int ret = mpiext_persistent_init_request(req);
	if (ret) {
		put_persistent_request(req->store_index);
		*request = NULL;
	}
	return ret;
}

int PRISM_Precv_init(void *buffer, int count, MPI_Datatype datatype, int src,
		     int tag, MPI_Comm communicator, PRISM_Request *request) {
	int size;
	assert(buffer != NULL);

	persistent_request_t *req = get_persistent_request();

	if (!req) return -FI_ENOMEM;

	req->req_comm = communicator;
	req->tag = tag;
	req->peer_rank = src;
	req->data_buffer = (void *)buffer;
	req->op_type = PERSISTENT_RECV;

	PMPI_Type_size(datatype, &size);
	req->size = (size_t)size * count;

	*request = (void *)req;
	OPT_PERSISTENT_INFO("[PRECV_INIT]: data buffer %p", (void *)buffer);
	int ret = mpiext_persistent_init_request(req);
	if (ret) {
		put_persistent_request(req->store_index);
		*request = NULL;
	}
	return ret;
}

static int mpiext_persistent_start_send_internal_no_sync(
    persistent_request_t *request) {
	if (PERSISTENT_LIKELY(request->init_state == READY))
		return mpiext_persistent_start_send_core_no_sync(request);
	else {
		return mpiext_persistent_finialize_init_first_no_sync(request);
	}
}

static int mpiext_persistent_start_recv_internal_no_sync(
    persistent_request_t *request) {
	if (PERSISTENT_LIKELY(request->init_state == READY))
		return mpiext_persistent_start_recv_core_no_sync(request);
	else {
		return mpiext_persistent_finialize_init_first_no_sync(request);
	}
}

static int mpiext_persistent_start_send_internal(
    persistent_request_t *request) {
	if (PERSISTENT_LIKELY(request->init_state == READY))
		return mpiext_persistent_start_send_core(request);
	else {
		return mpiext_persistent_finialize_init_first(request);
	}
}

static int mpiext_persistent_start_recv_internal(
    persistent_request_t *request) {
	if (PERSISTENT_LIKELY(request->init_state == READY))
		return mpiext_persistent_start_recv_core(request);
	else {
		return mpiext_persistent_finialize_init_first(request);
	}
}

static int mpiext_persistent_start_internal(persistent_request_t *request) {
	if (request->op_type == PERSISTENT_SEND)
		return mpiext_persistent_start_send_internal(request);
	else
		return mpiext_persistent_start_recv_internal(request);
}

static int mpiext_persistent_start_internal_no_sync(
    persistent_request_t *request) {
	if (request->op_type == PERSISTENT_SEND)
		return mpiext_persistent_start_send_internal_no_sync(request);
	else
		return mpiext_persistent_start_recv_internal_no_sync(request);
}

int PRISM_Start_no_sync(PRISM_Request *request) {
	return mpiext_persistent_start_internal_no_sync(
	    (persistent_request_t *)*request);
}

int PRISM_Startall_no_sync(int count, PRISM_Request request_array[]) {
	int ret = PRISM_ERROR;

	for (int i = 0; i < count; i++) {
		ret = mpiext_persistent_start_internal_no_sync(
		    (persistent_request_t *)*(request_array + i));
		if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
			return PRISM_ERROR;
		}
	}
	return PRISM_SUCCESS;
}

int PRISM_Start(PRISM_Request *request) {
	return mpiext_persistent_start_internal(
	    (persistent_request_t *)*request);
}

int PRISM_Startall(int count, PRISM_Request request_array[]) {
	int ret = PRISM_ERROR;

	for (int i = 0; i < count; i++) {
		if (!request_array[i]) continue;
		ret = mpiext_persistent_start_internal(
		    (persistent_request_t *)*(request_array + i));

		if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
			return PRISM_ERROR;
		}
	}
	return PRISM_SUCCESS;
}

static int mpiext_persistent_wait_internal(persistent_request_t *request) {
	if (request->op_type == PERSISTENT_SEND)
		return mpiext_persistent_finalize_send_core(request);
	else
		return mpiext_persistent_finalize_recv_core(request);
}

int PRISM_Wait(PRISM_Request *request, MPI_Status *status) {
	return mpiext_persistent_wait_internal(
	    (persistent_request_t *)*request);
}

int PRISM_Waitall(int count, PRISM_Request request_array[],
		  MPI_Status status_array[]) {
	int ret = PRISM_ERROR;

	for (int i = 0; i < count; i++) {
		if (!request_array[i]) continue;
		ret = mpiext_persistent_wait_internal(
		    (persistent_request_t *)*(request_array + i));
		if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
			return PRISM_ERROR;
		}
	}
	return PRISM_SUCCESS;
}

int PRISM_Prequest_free(PRISM_Request *request) {
	if (request == NULL || *request == NULL) return PRISM_SUCCESS;

	persistent_request_t *req = (persistent_request_t *)*request;

	put_persistent_request(req->store_index);

	*request = NULL;

	return PRISM_SUCCESS;
}
