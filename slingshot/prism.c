#include "prism.h"

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <mpi.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>
#include <fi_cxi_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PRISM_ERROR -1
#define PRISM_SUCCESS 0

// #define DEBUG
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

#define PERSISTENT_LIKELY(x) __builtin_expect(!!(x), 1)
#define PERSISTENT_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define PERSISTENT_ALWAYS_INLINE __attribute__((always_inline))

#define DEFAULT_REQUEST_STORE_SIZE UINT32_C(512)
#define PERSISTENT_EAGER_MAX 4096

static size_t persistent_eager_max = PERSISTENT_EAGER_MAX;

/* Setup remains on the application's MPI channel. All peers must use this
 * wire format; the marker also rejects old, direction-less setup records. */
#define PERSISTENT_SETUP_WORDS 11
#define PERSISTENT_SETUP_MAGIC UINT64_C(0x5052534358490004)

/* Each buffered receiver owns two independently counted target MRs. */
#define PERSISTENT_MR_KEYS (2 * DEFAULT_REQUEST_STORE_SIZE)
static bool remote_mr_keys[PERSISTENT_MR_KEYS];

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

typedef struct persistent_setup_record {
	uint64_t words[PERSISTENT_SETUP_WORDS];
	struct persistent_setup_record *next;
} persistent_setup_record_t;

typedef struct persistent_setup_block {
	persistent_setup_record_t records[DEFAULT_REQUEST_STORE_SIZE];
	struct persistent_setup_block *next;
} persistent_setup_block_t;

typedef struct persistent_setup_stream {
	MPI_Comm comm;
	int peer_rank;
	int tag;
	uint64_t
	    next_sequence[2]; /* Local SEND and RECV initialization order. */
	persistent_setup_record_t *pending;
	struct persistent_setup_stream *next;
} persistent_setup_stream_t;

typedef struct mpiext_persistent_ctx {
	struct fi_info *fi;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_cq *cq;
	struct fid_av *av;
	struct fid_ep *ep;
	struct fid_cntr *tx_cntr;
	uint64_t tx_posted;
	int completion_mode; /* 0: automatic, 1: ct, 2: ct-buffered, 3: ct-triggered */
	bool triggered;
	size_t trigger_capacity;
	size_t trigger_requests;
	struct persistent_request *pending_triggers;
	fi_addr_t *peer_addr;
	int n_peers;
	int my_world_rank;
	int error;
} mpiext_persistent_ctx_t;

typedef struct persistent_request {
	uint64_t remote_data_addr;
	uint64_t remote_flag_addr;
	uint64_t remote_data_mkey;
	uint64_t remote_flag_mkey;
	uint64_t posted_ops;
	uint64_t expected_rtr;
	uint64_t triggered_posted;
	uint64_t remote_second_mkey;
	struct fid_cntr *target_cntr;
	struct fid_cntr *second_cntr;
	struct fid_cntr *trigger_cntr;
	struct fid_mr *second_buffer_mr;
	uint32_t local_mr_key[2];
	bool active;
	bool trigger_pending;
	struct persistent_request *next_trigger;
	/* Keep deferred-work descriptors alive until completion. */
	struct iovec trigger_iov;
	struct fi_rma_iov trigger_rma_iov;
	struct fi_op_rma trigger_rma;
	struct fi_deferred_work trigger_work;
	size_t size;
	size_t my_rdma_info_size;
	struct fid_mr *data_buffer_mr;
	struct fid_mr *flag_buffer_mr;
	void *data_buffer;
	void *my_rdma_info_buffer;
	volatile uint64_t flag_buffer;
	int tag;
	int peer_rank;
	int store_index;
	MPI_Comm req_comm;
	mpiext_persistent_request_init_state_t init_state;
	mpiext_persistent_op_type_t op_type;
	request_store_state_t req_state;
	int error;
	fi_addr_t peer_addr;
	persistent_setup_stream_t *setup_stream;
	uint64_t setup_sequence;
	bool buffered;
	void *bounce_buffer;
	size_t remote_size;
	uint64_t buffered_sent;
	uint64_t buffered_consumed;
} persistent_request_t;

typedef struct request_element {
	uint32_t index;
	struct request_element *next;
} request_element_t;

static mpiext_persistent_ctx_t ctx;

static persistent_request_t *request_store = NULL;
static request_element_t *free_requests_head = NULL;
static request_element_t *free_requests_tail = NULL;
static request_element_t *request_element_mpool = NULL;
static persistent_setup_stream_t *setup_streams = NULL;
static persistent_setup_block_t *setup_blocks = NULL;
static persistent_setup_record_t *free_setup_records = NULL;

/* Independent of request storage: start with the same capacity and grow in
 * equally sized blocks if unmatched remote metadata exceeds that capacity. */
static int mpiext_persistent_grow_setup_pool(void) {
	persistent_setup_block_t *block = malloc(sizeof(*block));
	if (block == NULL) return MPI_ERR_NO_MEM;
	block->next = setup_blocks;
	setup_blocks = block;
	for (uint32_t i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
		block->records[i].next = free_setup_records;
		free_setup_records = &block->records[i];
	}
	return PRISM_SUCCESS;
}

static inline persistent_setup_record_t *mpiext_persistent_get_setup_record(
    void) {
	if (free_setup_records == NULL &&
	    mpiext_persistent_grow_setup_pool() != PRISM_SUCCESS)
		return NULL;
	persistent_setup_record_t *record = free_setup_records;
	free_setup_records = record->next;
	record->next = NULL;
	return record;
}

static inline void mpiext_persistent_put_setup_record(
    persistent_setup_record_t *record) {
	record->next = free_setup_records;
	free_setup_records = record;
}

static void mpiext_persistent_forget_setup(persistent_request_t *request) {
	if (request->setup_stream == NULL) return;
	uint64_t remote_type = request->op_type == PERSISTENT_SEND
				   ? PERSISTENT_RECV
				   : PERSISTENT_SEND;
	persistent_setup_record_t **link = &request->setup_stream->pending;
	while (*link != NULL) {
		persistent_setup_record_t *record = *link;
		if (record->words[5] == remote_type &&
		    record->words[6] == request->setup_sequence) {
			*link = record->next;
			mpiext_persistent_put_setup_record(record);
		} else {
			link = &record->next;
		}
	}
}

static int mpiext_persistent_get_setup_stream(
    persistent_request_t *request, persistent_setup_stream_t **result) {
	for (persistent_setup_stream_t *stream = setup_streams; stream;
	     stream = stream->next) {
		if (stream->comm == request->req_comm &&
		    stream->peer_rank == request->peer_rank &&
		    stream->tag == request->tag) {
			*result = stream;
			return PRISM_SUCCESS;
		}
	}

	persistent_setup_stream_t *stream = calloc(1, sizeof(*stream));
	if (stream == NULL) return MPI_ERR_NO_MEM;
	stream->comm = request->req_comm;
	stream->peer_rank = request->peer_rank;
	stream->tag = request->tag;
	stream->next = setup_streams;
	setup_streams = stream;
	*result = stream;
	return PRISM_SUCCESS;
}

static void mpiext_persistent_cleanup_setup(void) {
	while (setup_streams != NULL) {
		persistent_setup_stream_t *stream = setup_streams;
		setup_streams = stream->next;
		while (stream->pending != NULL) {
			persistent_setup_record_t *record = stream->pending;
			stream->pending = record->next;
			mpiext_persistent_put_setup_record(record);
		}
		free(stream);
	}
	free_setup_records = NULL;
	while (setup_blocks != NULL) {
		persistent_setup_block_t *block = setup_blocks;
		setup_blocks = block->next;
		free(block);
	}
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_cleanup_request_ressources(persistent_request_t *req) {
	mpiext_persistent_forget_setup(req);
	if (req->data_buffer_mr) fi_close(&req->data_buffer_mr->fid);
	if (req->flag_buffer_mr) fi_close(&req->flag_buffer_mr->fid);
	if (req->second_buffer_mr) fi_close(&req->second_buffer_mr->fid);
	if (req->target_cntr) fi_close(&req->target_cntr->fid);
	if (req->second_cntr) fi_close(&req->second_cntr->fid);
	if (req->trigger_cntr) {
		fi_close(&req->trigger_cntr->fid);
		ctx.trigger_requests--;
	}
	for (unsigned i = 0; i < 2; ++i)
		if (req->local_mr_key[i] < PERSISTENT_MR_KEYS)
			remote_mr_keys[req->local_mr_key[i]] = false;
	free(req->bounce_buffer);
	free(req->my_rdma_info_buffer);
}

static inline void free_requests_append(request_element_t *elem) {
	elem->next = NULL;
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
static inline void cleanup_request_storage(void);

static inline int init_request_storage(void) {
	request_element_t *elem = NULL;

	request_store =
	    malloc(DEFAULT_REQUEST_STORE_SIZE * sizeof(persistent_request_t));
	if (!request_store) {
		OPT_PERSISTENT_ERR("request_store alloc failed");
		return MPI_ERR_NO_MEM;
	}

	for (uint32_t i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
		request_store[i].req_state = REQ_FREE;
		request_store[i].store_index = i;
		mpiext_persistent_reset_request(&request_store[i]);
	}

	for (uint32_t i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
		elem = malloc(sizeof(request_element_t));

		if (!elem) {
			OPT_PERSISTENT_ERR("elem alloc failed");
			cleanup_request_storage();
			return MPI_ERR_NO_MEM;
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

	for (uint32_t i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
		if (request_store[i].req_state == REQ_FREE) continue;

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
	elem->next = NULL;
	return elem;
}

static inline persistent_request_t *get_persistent_request(void) {
	request_element_t *elem = free_requests_head;
	if (PERSISTENT_UNLIKELY(elem == NULL)) return NULL;
	free_requests_head = elem->next;
	if (free_requests_head == NULL) free_requests_tail = NULL;
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
	int index = request->store_index;
	request_store_state_t state = request->req_state;
	memset(request, 0, sizeof(*request));
	request->store_index = index;
	request->req_state = state;
	request->local_mr_key[0] = request->local_mr_key[1] = UINT32_MAX;
	request->tag = -1;
	request->req_comm = MPI_COMM_NULL;
	request->peer_rank = -1;
	request->peer_addr = FI_ADDR_NOTAVAIL;
	request->init_state = CHECK_FOR_REMOTE_INFO;
	request->op_type = OP_INVALID;
}

static inline int mpiext_persistent_get_ep(int target_rank, fi_addr_t *address) {
	*address = FI_ADDR_NOTAVAIL;
	if (target_rank < 0 || target_rank >= ctx.n_peers) return MPI_ERR_RANK;
	*address = ctx.peer_addr[target_rank];
	return PRISM_SUCCESS;
}

static void mpiext_persistent_cleanup_module(void);

int PRISM_Init(void) {
	int ret;
	struct fi_info *hints = NULL;
	struct fi_cq_attr cq_attr = {0};
	struct fi_av_attr av_attr = {0};
	char addr[64] = {0};
	size_t addr_len = sizeof(addr);
	int my_rank, world_size;
	const char *limit = getenv("PRISM_CXI_EAGER_LIMIT");
	int invalid_limit = 0, any_invalid_limit;

	persistent_eager_max = PERSISTENT_EAGER_MAX;
	if (limit) {
		char *end;
		errno = 0;
		unsigned long long value = strtoull(limit, &end, 10);
		if (errno || limit[0] < '0' || limit[0] > '9' || *end || value > SIZE_MAX)
			invalid_limit = 1;
		else
			persistent_eager_max = (size_t)value;
	}
	PMPI_Allreduce(&invalid_limit, &any_invalid_limit, 1, MPI_INT, MPI_MAX,
	               MPI_COMM_WORLD);
	if (any_invalid_limit) {
		OPT_PERSISTENT_ERR("PRISM_CXI_EAGER_LIMIT must be a nonnegative byte count");
		return MPI_ERR_ARG;
	}

	const char *mode = getenv("PRISM_CXI_COMPLETION");
	ctx.completion_mode = !mode || strcmp(mode, "auto") == 0 ? 0 :
		strcmp(mode, "ct") == 0 ? 1 :
		strcmp(mode, "ct-buffered") == 0 ? 2 :
		strcmp(mode, "ct-triggered") == 0 ? 3 : -1;
	int min_mode, max_mode;
	PMPI_Allreduce(&ctx.completion_mode, &min_mode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	PMPI_Allreduce(&ctx.completion_mode, &max_mode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (min_mode < 0 || min_mode != max_mode) {
		OPT_PERSISTENT_ERR("PRISM_CXI_COMPLETION must be auto, ct, ct-buffered or ct-triggered and identical on all ranks");
		return MPI_ERR_ARG;
	}
	ctx.triggered = ctx.completion_mode == 3;

	PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

	hints = fi_allocinfo();
	if (!hints) { ret = -FI_ENOMEM; goto local_setup_done; }

	hints->ep_attr->type = FI_EP_RDM;
	hints->fabric_attr->prov_name = strdup("cxi");
	if (!hints->fabric_attr->prov_name) { ret = -FI_ENOMEM; goto local_setup_done; }
	hints->caps = FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_RMA_EVENT;
	if (ctx.triggered) hints->caps |= FI_TRIGGER;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->mode = 0;
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
	hints->tx_attr->op_flags = 0;
	/* Credits overwrite one polled word: preserve their write order. */
	hints->tx_attr->msg_order = FI_ORDER_RMA_WAW;
	hints->addr_format = FI_FORMAT_UNSPEC;

	ret = fi_getinfo(FI_VERSION(2, 6), NULL, NULL, 0, hints, &ctx.fi);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}
	if (ctx.fi->tx_attr->inject_size < sizeof(uint64_t)) {
		ret = -FI_ENOSYS;
		goto local_setup_done;
	}
	fi_freeinfo(hints);
	hints = NULL;

#ifdef DEBUG
	if (my_rank == 0) {
		mpiext_persistent_print_provider_info(ctx.fi);
	}
#endif

	ret = fi_fabric(ctx.fi->fabric_attr, &ctx.fabric, NULL);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

	ret = fi_domain(ctx.fabric, ctx.fi, &ctx.domain, NULL);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

	/* Required by CXI for provider progress/errors; no success CQ events. */
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;

	ret = fi_cq_open(ctx.domain, &cq_attr, &ctx.cq, NULL);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

	av_attr.type = FI_AV_TABLE;
	av_attr.count = world_size;

	ctx.n_peers = world_size;

	ret = fi_av_open(ctx.domain, &av_attr, &ctx.av, NULL);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

	ret = fi_endpoint(ctx.domain, ctx.fi, &ctx.ep, NULL);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}
	ret = fi_ep_bind(ctx.ep, &ctx.cq->fid,
			 FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

	ret = fi_ep_bind(ctx.ep, &ctx.av->fid, 0);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

	struct fi_cntr_attr cntr_attr = {
		.events = FI_CNTR_EVENTS_COMP, .wait_obj = FI_WAIT_NONE,
	};
	ret = fi_cntr_open(ctx.domain, &cntr_attr, &ctx.tx_cntr, NULL);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}
	ret = fi_ep_bind(ctx.ep, &ctx.tx_cntr->fid, FI_WRITE);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

	/* Enable the endpoint: from this point data-path calls are valid. */
	ret = fi_enable(ctx.ep);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI initialization failed: %s", fi_strerror(-ret));
		goto local_setup_done;
	}

local_setup_done:;
	/* All ranks reach the same failure boundary before address exchange. */
	int failed = ret != 0, any_failed;
	PMPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (any_failed) goto fail;

	if (ctx.triggered) {
		/* DWQ resources belong to the service/NIC, not to this domain.
		 * Reserve a conservative equal share for each node-local rank. */
		struct fi_cxi_dom_ops *ops = NULL;
		size_t depth = 0;
		MPI_Comm local_comm;
		int local_size;
		ret = PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
		                          my_rank, MPI_INFO_NULL, &local_comm);
		if (ret != MPI_SUCCESS) goto fail;
		PMPI_Comm_size(local_comm, &local_size);
		PMPI_Comm_free(&local_comm);
		ret = fi_open_ops(&ctx.domain->fid, FI_CXI_DOM_OPS_4, 0,
		                  (void **)&ops, NULL);
		if (!ret) ret = ops->get_dwq_depth(&ctx.domain->fid, &depth);
		if (!ret && depth > 8) ctx.trigger_capacity = (depth - 8) / local_size;
		int unavailable = ret != 0 || ctx.trigger_capacity == 0, any_unavailable;
		PMPI_Allreduce(&unavailable, &any_unavailable, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		if (any_unavailable) { ret = MPI_ERR_OTHER; goto fail; }
	}
	/* Exchange endpoint names in world-rank order, as in the OPA backend.
	 * Use fixed, zero-padded records so every rank has the same MPI count. */
	ret = fi_getname(&ctx.ep->fid, addr, &addr_len);
	failed = ret != 0 || addr_len > sizeof(addr);
	PMPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (any_failed) { ret = MPI_ERR_OTHER; goto fail; }

	ctx.peer_addr = calloc(ctx.n_peers, sizeof(fi_addr_t));
	char *addresses = malloc((size_t)world_size * sizeof(addr));
	failed = ctx.peer_addr == NULL || addresses == NULL;
	PMPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (any_failed) {
		free(addresses);
		free(ctx.peer_addr);
		ctx.peer_addr = NULL;
		ret = MPI_ERR_NO_MEM;
		goto fail;
	}
	ret = PMPI_Allgather(addr, sizeof(addr), MPI_BYTE, addresses,
	                    sizeof(addr), MPI_BYTE, MPI_COMM_WORLD);
	if (ret != MPI_SUCCESS) {
		free(addresses);
		goto fail;
	}
	failed = 0;
	for (int i = 0; i < world_size; ++i) {
		ret = fi_av_insert(ctx.av, addresses + (size_t)i * sizeof(addr),
		                   1, &ctx.peer_addr[i], 0, NULL);
		if (ret != 1) failed = 1;
	}
	free(addresses);
	PMPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (any_failed) {
		OPT_PERSISTENT_ERR("fi_av_insert failed");
		ret = MPI_ERR_OTHER;
		goto fail;
	}

	ctx.my_world_rank = my_rank;
	ret = init_request_storage();
	if (ret == PRISM_SUCCESS) ret = mpiext_persistent_grow_setup_pool();
	failed = ret != PRISM_SUCCESS;
	PMPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (!any_failed) return PRISM_SUCCESS;
fail:
	if (request_store) cleanup_request_storage();
	mpiext_persistent_cleanup_setup();
	if (hints) fi_freeinfo(hints);
	mpiext_persistent_cleanup_module();
	return MPI_ERR_OTHER;
}

static inline int mpiext_persistent_register_memory(
    persistent_request_t *request) {
	struct fi_cntr_attr attr = {
		.events = FI_CNTR_EVENTS_COMP, .wait_obj = FI_WAIT_NONE,
	};
	unsigned targets = request->buffered && request->op_type == PERSISTENT_RECV ? 2 : 1;
	size_t stride = request->size ? request->size : 1;
	int ret;
	if (request->size > ctx.fi->ep_attr->max_msg_size) return MPI_ERR_COUNT;
	if (targets == 2) {
		if (stride > SIZE_MAX / 2) return MPI_ERR_COUNT;
		if (posix_memalign(&request->bounce_buffer, 64, 2 * stride))
			return MPI_ERR_NO_MEM;
	}
	if (ctx.triggered && request->op_type == PERSISTENT_SEND) {
		if (ctx.trigger_requests >= ctx.trigger_capacity) {
			OPT_PERSISTENT_ERR("CXI triggered request capacity exhausted (%zu)", ctx.trigger_capacity);
			return MPI_ERR_NO_MEM;
		}
		ret = fi_cntr_open(ctx.domain, &attr, &request->trigger_cntr, NULL);
		if (ret) return MPI_ERR_OTHER;
		ctx.trigger_requests++;
		/* Software advances this counter after polling RTR. It is never
		 * associated with the flag MR. */
		ret = fi_cntr_open(ctx.domain, &attr, &request->target_cntr, NULL);
		if (ret) goto fail;
	}
	for (unsigned slot = 0; slot < targets; ++slot) {
		struct fid_cntr **cntr = slot ? &request->second_cntr : &request->target_cntr;
		struct fid_mr **mr = request->op_type == PERSISTENT_SEND ? &request->flag_buffer_mr :
			slot ? &request->second_buffer_mr : &request->data_buffer_mr;
		void *buffer = request->op_type == PERSISTENT_SEND ? (void *)&request->flag_buffer :
			request->bounce_buffer ? (char *)request->bounce_buffer + slot * stride : request->data_buffer;
		size_t length = request->op_type == PERSISTENT_SEND ? sizeof(request->flag_buffer) : stride;
		uint32_t key;
		for (key = 0; key < PERSISTENT_MR_KEYS && remote_mr_keys[key]; ++key) {}
		if (key == PERSISTENT_MR_KEYS) return MPI_ERR_NO_MEM;
		remote_mr_keys[key] = true;
		request->local_mr_key[slot] = key;
		bool counted = request->op_type == PERSISTENT_RECV;
		if (counted) {
			ret = fi_cntr_open(ctx.domain, &attr, cntr, NULL);
			if (ret) goto fail;
		}
		ret = fi_mr_reg(ctx.domain, buffer, length, FI_REMOTE_WRITE, 0, key,
		                counted ? FI_RMA_EVENT : 0, mr, NULL);
		if (ret) goto fail;
		ret = fi_mr_bind(*mr, &ctx.ep->fid, 0);
		if (ret) goto fail;
		if (counted) {
			ret = fi_mr_bind(*mr, &(*cntr)->fid, FI_REMOTE_WRITE);
			if (ret) goto fail;
		}
		ret = fi_mr_enable(*mr);
		if (ret) goto fail;
	}
	return PRISM_SUCCESS;
fail:
	OPT_PERSISTENT_ERR("CXI target MR/counter setup failed: %s", fi_strerror(-ret));
	/* init_request's failure path releases partial resources and keys. */
	return MPI_ERR_OTHER;
}

PERSISTENT_ALWAYS_INLINE static inline void mpiext_persistent_pack_rdma_info(
    persistent_request_t *request) {
	uint64_t *packed = request->my_rdma_info_buffer;
	packed[0] = 0; /* CXI uses offsets relative to the target MR. */
	packed[1] = request->data_buffer_mr ? fi_mr_key(request->data_buffer_mr) : FI_KEY_NOTAVAIL;
	packed[2] = 0;
	packed[3] = request->flag_buffer_mr ? fi_mr_key(request->flag_buffer_mr) : FI_KEY_NOTAVAIL;
	packed[4] = (uint64_t)request->store_index;
	packed[5] = (uint64_t)request->op_type;
	packed[6] = request->setup_sequence;
	packed[7] = PERSISTENT_SETUP_MAGIC;
	packed[8] = request->buffered;
	packed[9] = request->size;
	packed[10] = request->second_buffer_mr ? fi_mr_key(request->second_buffer_mr) : FI_KEY_NOTAVAIL;
}

PERSISTENT_ALWAYS_INLINE static inline int mpiext_persistent_unpack_rdma_info(
    persistent_request_t *request, void *payload) {
	uint64_t *packed = payload;
	if (packed[8] != (uint64_t)request->buffered) {
		OPT_PERSISTENT_ERR("Buffered mode must match on both peers");
		return request->error = MPI_ERR_ARG;
	}
	request->remote_size = packed[9];
	if ((request->op_type == PERSISTENT_SEND
				      ? request->size > request->remote_size
				      : request->remote_size > request->size))
		return request->error = MPI_ERR_TRUNCATE;
	request->remote_data_addr = packed[0];
	request->remote_data_mkey = packed[1];
	request->remote_flag_addr = packed[2];
	request->remote_flag_mkey = packed[3];
	request->remote_second_mkey = packed[10];
	return PRISM_SUCCESS;
}

static void mpiext_persistent_cleanup_module(void) {
	if (ctx.ep) fi_close(&ctx.ep->fid);
	if (ctx.tx_cntr) fi_close(&ctx.tx_cntr->fid);
	if (ctx.av) {
		fi_close(&ctx.av->fid);
	}
	if (ctx.cq) {
		fi_close(&ctx.cq->fid);
	}
	if (ctx.domain) {
		fi_close(&ctx.domain->fid);
	}
	if (ctx.fabric) {
		fi_close(&ctx.fabric->fid);
	}
	if (ctx.fi) {
		fi_freeinfo(ctx.fi);
	}
	free(ctx.peer_addr);
	memset(&ctx, 0, sizeof(ctx));
}

int PRISM_Finalize(void) {
	if (request_store) {
		for (unsigned i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i)
			if (request_store[i].req_state == REQ_ACTIVE && request_store[i].active)
				return MPI_ERR_REQUEST;
		for (unsigned i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
			if (request_store[i].req_state == REQ_FREE) continue;
			PRISM_Request request = &request_store[i];
			int ret = PRISM_Prequest_free(&request);
			if (ret != PRISM_SUCCESS) return ret;
		}
		cleanup_request_storage();
	}
	mpiext_persistent_cleanup_setup();
	mpiext_persistent_cleanup_module();
	return PRISM_SUCCESS;
}

static int mpiext_persistent_init_request(persistent_request_t *request) {
	persistent_setup_stream_t *stream;
	int ret = mpiext_persistent_register_memory(request);
	if (ret != PRISM_SUCCESS) goto fail;
	request->my_rdma_info_size = PERSISTENT_SETUP_WORDS * sizeof(uint64_t);

	request->my_rdma_info_buffer = malloc(request->my_rdma_info_size);
	if (request->my_rdma_info_buffer == NULL) {
		ret = MPI_ERR_NO_MEM;
		goto fail;
	}

	ret = mpiext_persistent_get_setup_stream(request, &stream);
	if (ret != PRISM_SUCCESS) goto fail;
	int direction = request->op_type - PERSISTENT_SEND;
	if (stream->next_sequence[direction] == UINT64_MAX) {
		ret = MPI_ERR_OTHER;
		goto fail;
	}
	request->setup_sequence = stream->next_sequence[direction];
	mpiext_persistent_pack_rdma_info(request);

	// We assume eager protocol usage here
	ret = PMPI_Send(request->my_rdma_info_buffer,
			request->my_rdma_info_size, MPI_BYTE,
			request->peer_rank, request->tag, request->req_comm);
	if (ret != PRISM_SUCCESS) goto fail;

	stream->next_sequence[direction]++;
	request->setup_stream = stream;
	return PRISM_SUCCESS;
fail:
	put_persistent_request(request->store_index);
	return ret;
}

static inline int mpiext_persistent_request_error(
    persistent_request_t *request) {
	return ctx.error != PRISM_SUCCESS ? ctx.error : request->error;
}

/* CQ access only drives CXI's manual progress and reports asynchronous errors.
 * Readiness is polled from flags; data completion uses counters. */
PERSISTENT_ALWAYS_INLINE static inline int mpiext_persistent_progress(void) {
	if (ctx.error != PRISM_SUCCESS) return ctx.error;
	struct fi_cq_entry unexpected;
	ssize_t ret = fi_cq_read(ctx.cq, &unexpected, 1);
	if (ret == -FI_EAVAIL) {
		struct fi_cq_err_entry err = {0};
		if (fi_cq_readerr(ctx.cq, &err, 0) == 1)
			OPT_PERSISTENT_ERR("CXI CQ error: %s", fi_cq_strerror(ctx.cq, err.prov_errno, err.err_data, NULL, 0));
		return ctx.error = MPI_ERR_OTHER;
	}
	/* Successful entries were suppressed at endpoint binding/submission. */
	if (ret > 0 || (ret < 0 && ret != -FI_EAGAIN))
		return ctx.error = MPI_ERR_OTHER;
	if (fi_cntr_readerr(ctx.tx_cntr)) {
		OPT_PERSISTENT_ERR("CXI transmit counter failure");
		return ctx.error = MPI_ERR_OTHER;
	}
	/* Service every queued sender, including when the caller waits for a
	 * receive or a different send. RTR flag MRs have no bound counter. */
	persistent_request_t **link = &ctx.pending_triggers;
	while (*link) {
		persistent_request_t *request = *link;
		if (request->flag_buffer < request->expected_rtr) {
			link = &request->next_trigger;
			continue;
		}
		int rc = fi_cntr_set(request->target_cntr, request->expected_rtr);
		if (rc == -FI_EAGAIN) {
			link = &request->next_trigger;
			continue;
		}
		if (rc) return ctx.error = request->error = MPI_ERR_OTHER;
		*link = request->next_trigger;
		request->next_trigger = NULL;
	}
	return PRISM_SUCCESS;
}

static inline int mpiext_persistent_progress_request(persistent_request_t *request) {
	int ret = mpiext_persistent_progress();
	if (ret != PRISM_SUCCESS) return ret;
	if ((request->target_cntr && fi_cntr_readerr(request->target_cntr)) ||
	    (request->second_cntr && fi_cntr_readerr(request->second_cntr)) ||
	    (request->trigger_cntr && fi_cntr_readerr(request->trigger_cntr))) {
		OPT_PERSISTENT_ERR("CXI request counter failure (request %d)",
		                   request->store_index);
		request->error = MPI_ERR_OTHER;
	}
	return mpiext_persistent_request_error(request);
}

/* CXI consumes the descriptors at submission, but retain them with the request
 * until Wait for the deferred-work interface's lifetime requirements. */
static int mpiext_persistent_start_triggered_transfer(persistent_request_t *request) {
	if (request->expected_rtr == FI_CXI_CNTR_SUCCESS_MAX ||
	    request->triggered_posted == FI_CXI_CNTR_SUCCESS_MAX)
		return request->error = MPI_ERR_OTHER;
	request->trigger_iov = (struct iovec){request->data_buffer, request->size};
	request->trigger_rma_iov = (struct fi_rma_iov){
		.addr = request->remote_data_addr, .len = request->size,
		.key = request->remote_data_mkey,
	};
	request->trigger_rma = (struct fi_op_rma){
		.ep = ctx.ep,
		.msg = {.msg_iov = &request->trigger_iov, .iov_count = 1,
		        .addr = request->peer_addr, .rma_iov = &request->trigger_rma_iov,
		        .rma_iov_count = 1, .context = request},
	};
	request->trigger_work = (struct fi_deferred_work){
		.threshold = request->expected_rtr + 1,
		.triggering_cntr = request->target_cntr,
		.completion_cntr = request->trigger_cntr,
		.op_type = FI_OP_WRITE, .op.rma = &request->trigger_rma,
	};
	int ret;
	do {
		ret = fi_control(&ctx.domain->fid, FI_QUEUE_WORK, &request->trigger_work);
		if (ret == -FI_EAGAIN) {
			int progress = mpiext_persistent_progress_request(request);
			if (progress != PRISM_SUCCESS) return progress;
		}
	} while (ret == -FI_EAGAIN);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI FI_QUEUE_WORK failed: %s", fi_strerror(-ret));
		return request->error = MPI_ERR_OTHER;
	}
	request->expected_rtr++;
	request->triggered_posted++;
	request->trigger_pending = true;
	request->next_trigger = ctx.pending_triggers;
	ctx.pending_triggers = request;
	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_data_transfer(persistent_request_t *request) {
	if (ctx.tx_posted == FI_CXI_CNTR_SUCCESS_MAX)
		return ctx.error = MPI_ERR_OTHER;
	unsigned slot = request->buffered ? request->buffered_sent % 2 : 0;
	struct fi_rma_iov remote = {
		.addr = request->remote_data_addr, .len = request->size,
		.key = slot ? request->remote_second_mkey : request->remote_data_mkey,
	};
	struct iovec iov = {request->data_buffer, request->size};
	struct fi_msg_rma msg = {
		.msg_iov = &iov, .iov_count = 1, .addr = request->peer_addr,
		.rma_iov = &remote, .rma_iov_count = 1, .context = request,
	};
	/* CXI does not require FI_MR_LOCAL. NULL descriptors let the provider
	 * manage source registration. Injection still increments the TX counter. */
	uint64_t flags = request->size <= ctx.fi->tx_attr->inject_size ? FI_INJECT : 0;
	ssize_t ret;
	do {
		ret = fi_writemsg(ctx.ep, &msg, flags);
		if (ret == -FI_EAGAIN) {
			int progress = mpiext_persistent_progress_request(request);
			if (progress != PRISM_SUCCESS) return progress;
		}
	} while (ret == -FI_EAGAIN);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI data write failed: %s", fi_strerror(-ret));
		return request->error = MPI_ERR_OTHER;
	}
	request->posted_ops = ++ctx.tx_posted;
	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_flag_transfer(persistent_request_t *request) {
	if (ctx.tx_posted == FI_CXI_CNTR_SUCCESS_MAX)
		return ctx.error = MPI_ERR_OTHER;
	if (request->flag_buffer == FI_CXI_CNTR_SUCCESS_MAX)
		return request->error = MPI_ERR_OTHER;
	/* Inject a monotonically increasing RTR/credit sequence. Injection
	 * copies the source before return; the target is polled without reset. */
	uint64_t value = request->flag_buffer + 1;
	struct iovec iov = {&value, sizeof(value)};
	struct fi_rma_iov remote = {
		.addr = request->remote_flag_addr, .len = sizeof(value),
		.key = request->remote_flag_mkey,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov, .iov_count = 1, .addr = request->peer_addr,
		.rma_iov = &remote, .rma_iov_count = 1, .context = request,
	};
	ssize_t ret;
	do {
		ret = fi_writemsg(ctx.ep, &msg, FI_INJECT);
		if (ret == -FI_EAGAIN) {
			int progress = mpiext_persistent_progress_request(request);
			if (progress != PRISM_SUCCESS) return progress;
		}
	} while (ret == -FI_EAGAIN);
	if (ret) {
		OPT_PERSISTENT_ERR("CXI control write failed: %s", fi_strerror(-ret));
		return request->error = MPI_ERR_OTHER;
	}
	request->flag_buffer = value;
	ctx.tx_posted++;
	return PRISM_SUCCESS;
}

static inline int mpiext_persistent_start_send_core_no_sync(persistent_request_t *request) {
	return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_start_send_core(persistent_request_t *request) {
	if (ctx.triggered) return mpiext_persistent_start_triggered_transfer(request);
	if (request->expected_rtr == FI_CXI_CNTR_SUCCESS_MAX)
		return request->error = MPI_ERR_OTHER;
	uint64_t threshold = request->expected_rtr + 1;
	do {
		int ret = mpiext_persistent_progress_request(request);
		if (ret != PRISM_SUCCESS) return ret;
	} while (request->flag_buffer < threshold);
	request->expected_rtr = threshold;
	return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_finalize_send_core(persistent_request_t *request) {
	/* An endpoint has one FI_WRITE counter, and operations may complete out
	 * of order. Drain the complete set of posted immediate writes, not a
	 * request's posting ordinal. Deferred writes have independent counters
	 * so an unrelated, unmet trigger cannot block this wait. */
	uint64_t threshold = request->trigger_pending ? request->triggered_posted : ctx.tx_posted;
	struct fid_cntr *cntr = request->trigger_pending ? request->trigger_cntr : ctx.tx_cntr;
	do {
		int ret = mpiext_persistent_progress_request(request);
		if (ret != PRISM_SUCCESS) return ret;
	} while (fi_cntr_read(cntr) < threshold);
	request->trigger_pending = false;
	return mpiext_persistent_progress_request(request);
}

static inline int mpiext_persistent_start_recv_core_no_sync(persistent_request_t *request) {
	request->posted_ops++;
	return PRISM_SUCCESS;
}

static inline int mpiext_persistent_start_recv_core(persistent_request_t *request) {
	int ret = mpiext_persistent_start_flag_transfer(request);
	if (ret == PRISM_SUCCESS) request->posted_ops++;
	return ret;
}

static inline int mpiext_persistent_finalize_recv_core(persistent_request_t *request) {
	do {
		int ret = mpiext_persistent_progress_request(request);
		if (ret != PRISM_SUCCESS) return ret;
	} while (fi_cntr_read(request->target_cntr) < request->posted_ops);
	return mpiext_persistent_progress_request(request);
}

/* True if this record belongs to an initialization not yet seen locally, or
 * to a live request. Discard late metadata for freed, unstarted requests. */
static bool mpiext_persistent_setup_needed(
    persistent_setup_stream_t *stream,
    const persistent_setup_record_t *record) {
	mpiext_persistent_op_type_t local_type =
	    record->words[5] == PERSISTENT_SEND ? PERSISTENT_RECV
						: PERSISTENT_SEND;
	if (record->words[6] >=
	    stream->next_sequence[local_type - PERSISTENT_SEND])
		return true;
	for (uint32_t i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
		persistent_request_t *candidate = &request_store[i];
		if (candidate->req_state == REQ_ACTIVE &&
		    candidate->setup_stream == stream &&
		    candidate->op_type == local_type &&
		    candidate->setup_sequence == record->words[6] &&
		    candidate->init_state == CHECK_FOR_REMOTE_INFO)
			return true;
	}
	return false;
}

static int mpiext_persistent_match_setup(persistent_request_t *request) {
	persistent_setup_stream_t *stream = request->setup_stream;
	uint64_t remote_type = request->op_type == PERSISTENT_SEND
				   ? PERSISTENT_RECV
				   : PERSISTENT_SEND;
	persistent_setup_record_t **link = &stream->pending;
	while (*link != NULL) {
		persistent_setup_record_t *record = *link;
		if (record->words[5] == remote_type &&
		    record->words[6] == request->setup_sequence) {
			*link = record->next;
			int ret = mpiext_persistent_unpack_rdma_info(
			    request, record->words);
			mpiext_persistent_put_setup_record(record);
			return ret;
		}
		link = &record->next;
	}

	for (;;) {
		/* Reserve a pooled record before consuming a message so pool
		 * growth failure cannot lose another request's metadata. */
		persistent_setup_record_t *record =
		    mpiext_persistent_get_setup_record();
		if (record == NULL) return request->error = MPI_ERR_NO_MEM;
		MPI_Status status;
		int ret = PMPI_Recv(record->words, sizeof(record->words),
				    MPI_BYTE, request->peer_rank, request->tag,
				    request->req_comm, &status);
		if (ret != PRISM_SUCCESS) {
			mpiext_persistent_put_setup_record(record);
			return ctx.error = ret;
		}
		int bytes;
		ret = PMPI_Get_count(&status, MPI_BYTE, &bytes);
		if (ret != PRISM_SUCCESS || bytes != sizeof(record->words) ||
		    record->words[7] != PERSISTENT_SETUP_MAGIC ||
		    (record->words[5] != PERSISTENT_SEND &&
		     record->words[5] != PERSISTENT_RECV) ||
		    record->words[4] >= DEFAULT_REQUEST_STORE_SIZE ||
		    record->words[6] == UINT64_MAX) {
			OPT_PERSISTENT_ERR(
			    "Invalid persistent setup record from rank %d, tag "
			    "%d",
			    request->peer_rank, request->tag);
			mpiext_persistent_put_setup_record(record);
			return ctx.error = MPI_ERR_OTHER;
		}
		if (record->words[5] == remote_type &&
		    record->words[6] == request->setup_sequence) {
			ret = mpiext_persistent_unpack_rdma_info(request,
								 record->words);
			mpiext_persistent_put_setup_record(record);
			return ret;
		}
		if (mpiext_persistent_setup_needed(stream, record)) {
			record->next = stream->pending;
			stream->pending = record;
		} else {
			mpiext_persistent_put_setup_record(record);
		}
	}
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first_no_sync(persistent_request_t *request) {
	int ret = mpiext_persistent_match_setup(request);
	if (ret != PRISM_SUCCESS) return ret;

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
	int ret = mpiext_persistent_match_setup(request);
	if (ret != PRISM_SUCCESS) return ret;

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

static inline bool mpiext_persistent_use_eager(size_t bytes) {
	return ctx.completion_mode == 2 ||
	       (ctx.completion_mode == 0 && bytes > ctx.fi->tx_attr->inject_size &&
	        bytes <= persistent_eager_max);
}

static int mpiext_persistent_message_size(int count, MPI_Datatype datatype,
					  size_t *bytes) {
	if (count < 0) return MPI_ERR_COUNT;
	if (datatype == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
	int size;
	int ret = PMPI_Type_size(datatype, &size);
	if (ret != MPI_SUCCESS) return ret;
	if (size < 0 || (size != 0 && (size_t)count > SIZE_MAX / (size_t)size))
		return MPI_ERR_COUNT;
	*bytes = (size_t)size * (size_t)count;
	return PRISM_SUCCESS;
}

/* A separate initialization API fixes the mode before addresses are exchanged.
 * Buffered requests subsequently use the ordinary Start/Wait entry points. */
static int mpiext_persistent_init_buffered(void *buffer, int count,
					   MPI_Datatype datatype, int peer,
					   int tag, MPI_Comm comm,
					   mpiext_persistent_op_type_t op,
					   PRISM_Request *result) {
	if (result == NULL) return MPI_ERR_ARG;
	*result = NULL;
	if (count < 0) return MPI_ERR_COUNT;
	if (buffer == NULL) return MPI_ERR_BUFFER;
	if (datatype == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
	int size;
	MPI_Aint lb, extent;
	int ret = PMPI_Type_size(datatype, &size);
	if (ret != MPI_SUCCESS) return ret;
	ret = PMPI_Type_get_true_extent(datatype, &lb, &extent);
	if (ret != MPI_SUCCESS) return ret;
	/* PRISM uses contiguous buffers; check the layout through public MPI. */
	if (lb != 0 || extent != size) return MPI_ERR_TYPE;
	ret = PMPI_Type_get_extent(datatype, &lb, &extent);
	if (ret != MPI_SUCCESS) return ret;
	if (count > 1 && extent != size) return MPI_ERR_TYPE;
	if (size < 0 ||
	    (size != 0 && (size_t)count > SIZE_MAX / (size_t)size / 2))
		return MPI_ERR_COUNT;

	persistent_request_t *request = get_persistent_request();
	if (request == NULL) return MPI_ERR_NO_MEM;
	ret = mpiext_persistent_get_ep(peer, &request->peer_addr);
	if (ret != PRISM_SUCCESS) {
		put_persistent_request(request->store_index);
		return ret;
	}
	request->req_comm = comm;
	request->peer_rank = peer;
	request->tag = tag;
	request->op_type = op;
	request->buffered = true;
	request->data_buffer = buffer;
	request->size = (size_t)size * (size_t)count;
	ret = mpiext_persistent_init_request(request);
	if (ret == PRISM_SUCCESS) *result = (PRISM_Request)request;
	return ret;
}

int PRISM_Psend_init(const void *buffer, int count, MPI_Datatype datatype,
		    int dst, int tag, MPI_Comm communicator,
		    PRISM_Request *request) {
	size_t bytes;

	assert(buffer != NULL);

	if (request == NULL) return MPI_ERR_ARG;
	*request = NULL;
	if (!ctx.fi || !ctx.domain || !ctx.ep || !request_store) {
		return MPI_ERR_OTHER;
	}
	int ret = mpiext_persistent_message_size(count, datatype, &bytes);
	if (ret != MPI_SUCCESS) return ret;
	/* Choose before registering memory or advertising the receive target.
	 * Paired automatic requests must specify equal byte counts. */
	if (mpiext_persistent_use_eager(bytes))
		return mpiext_persistent_init_buffered(
		    (void *)buffer, count, datatype, dst, tag, communicator,
		    PERSISTENT_SEND, request);
	persistent_request_t *req = get_persistent_request();

	if (req == NULL) return MPI_ERR_NO_MEM;

	ret = mpiext_persistent_get_ep(dst, &req->peer_addr);
	if (ret != PRISM_SUCCESS) {
		put_persistent_request(req->store_index);
		return ret;
	}

	req->req_comm = communicator;
	req->tag = tag;
	req->peer_rank = dst;
	req->data_buffer = (void *)buffer;
	req->op_type = PERSISTENT_SEND;

	req->size = bytes;

	OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p", (void *)buffer);
	ret = mpiext_persistent_init_request(req);
	if (ret == PRISM_SUCCESS) *request = (void *)req;
	return ret;
}

int PRISM_Precv_init(void *buffer, int count, MPI_Datatype datatype, int src,
		    int tag, MPI_Comm communicator, PRISM_Request *request) {
	size_t bytes;
	assert(buffer != NULL);

	if (request == NULL) return MPI_ERR_ARG;
	*request = NULL;
	if (!ctx.fi || !ctx.domain || !ctx.ep || !request_store) {
		return MPI_ERR_OTHER;
	}
	int ret = mpiext_persistent_message_size(count, datatype, &bytes);
	if (ret != MPI_SUCCESS) return ret;
	if (mpiext_persistent_use_eager(bytes))
		return mpiext_persistent_init_buffered(
		    (void *)buffer, count, datatype, src, tag, communicator,
		    PERSISTENT_RECV, request);

	persistent_request_t *req = get_persistent_request();

	if (req == NULL) return MPI_ERR_NO_MEM;

	ret = mpiext_persistent_get_ep(src, &req->peer_addr);
	if (ret != PRISM_SUCCESS) {
		put_persistent_request(req->store_index);
		return ret;
	}

	req->req_comm = communicator;
	req->tag = tag;
	req->peer_rank = src;
	req->data_buffer = buffer;
	req->op_type = PERSISTENT_RECV;

	req->size = bytes;

	OPT_PERSISTENT_INFO("[PRECV_INIT]: data buffer %p", (void *)buffer);
	ret = mpiext_persistent_init_request(req);
	if (ret == PRISM_SUCCESS) *request = (void *)req;
	return ret;
}



static int mpiext_persistent_start_buffered(persistent_request_t *request) {
	if (request->init_state != READY) {
		int ret = mpiext_persistent_match_setup(request);
		if (ret != PRISM_SUCCESS) return ret;
		request->init_state = READY;
	}
	if (request->op_type == PERSISTENT_SEND) {
		/* The first two slots are free. Credits count consumed messages;
		 * a later credit also proves every earlier message was consumed. */
		uint64_t needed = request->buffered_sent < 2 ? 0 : request->buffered_sent - 1;
		do {
			int ret = mpiext_persistent_progress_request(request);
			if (ret != PRISM_SUCCESS) return ret;
		} while (request->flag_buffer < needed);
		int ret = mpiext_persistent_start_data_transfer(request);
		if (ret != PRISM_SUCCESS) return ret;
		request->buffered_sent++;
	} else {
		request->posted_ops++;
	}
	return PRISM_SUCCESS;
}

static int mpiext_persistent_wait_buffered(persistent_request_t *request) {
	if (request->op_type == PERSISTENT_SEND)
		return mpiext_persistent_finalize_send_core(request);
	unsigned slot = request->buffered_consumed % 2;
	struct fid_cntr *cntr = slot ? request->second_cntr : request->target_cntr;
	uint64_t threshold = request->buffered_consumed / 2 + 1;
	do {
		int ret = mpiext_persistent_progress_request(request);
		if (ret != PRISM_SUCCESS) return ret;
	} while (fi_cntr_read(cntr) < threshold);
	memcpy(request->data_buffer,
	       (char *)request->bounce_buffer + slot * (request->size ? request->size : 1),
	       request->remote_size);
	int ret = mpiext_persistent_start_flag_transfer(request);
	if (ret != PRISM_SUCCESS) return ret;
	request->buffered_consumed++;
	return PRISM_SUCCESS;
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
	if (request->active) return MPI_ERR_REQUEST;
	if (request->posted_ops == FI_CXI_CNTR_SUCCESS_MAX ||
	    request->buffered_sent == FI_CXI_CNTR_SUCCESS_MAX)
		return request->error = MPI_ERR_OTHER;
	int ret = mpiext_persistent_request_error(request);
	if (PERSISTENT_UNLIKELY(ret != PRISM_SUCCESS)) return ret;
	request->active = true;
	if (request->buffered) return mpiext_persistent_start_buffered(request);
	if (request->op_type == PERSISTENT_SEND)
		return mpiext_persistent_start_send_internal(request);
	else
		return mpiext_persistent_start_recv_internal(request);
}

static int mpiext_persistent_start_internal_no_sync(
    persistent_request_t *request) {
	if (request->active) return MPI_ERR_REQUEST;
	if (request->posted_ops == FI_CXI_CNTR_SUCCESS_MAX ||
	    request->buffered_sent == FI_CXI_CNTR_SUCCESS_MAX)
		return request->error = MPI_ERR_OTHER;
	int ret = mpiext_persistent_request_error(request);
	if (PERSISTENT_UNLIKELY(ret != PRISM_SUCCESS)) return ret;
	request->active = true;
	if (request->buffered) return mpiext_persistent_start_buffered(request);
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
			return ret;
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
			return ret;
		}
	}
	return PRISM_SUCCESS;
}

static int mpiext_persistent_wait_internal(persistent_request_t *request) {
	int ret = mpiext_persistent_request_error(request);
	if (ret != PRISM_SUCCESS || !request->active) return ret;
	if (request->buffered) ret = mpiext_persistent_wait_buffered(request);
	else if (request->op_type == PERSISTENT_SEND) ret = mpiext_persistent_finalize_send_core(request);
	else ret = mpiext_persistent_finalize_recv_core(request);
	if (ret == PRISM_SUCCESS) request->active = false;
	return ret;
}

int PRISM_Wait(PRISM_Request *request, MPI_Status *status) {
	(void)status;
	return mpiext_persistent_wait_internal(
	    (persistent_request_t *)*request);
}

int PRISM_Waitall(int count, PRISM_Request request_array[],
		 MPI_Status status_array[]) {
	(void)status_array;
	int ret = PRISM_ERROR;

	for (int i = 0; i < count; i++) {
		if (!request_array[i]) continue;
		ret = mpiext_persistent_wait_internal(
		    (persistent_request_t *)*(request_array + i));
		if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
			return ret;
		}
	}
	return PRISM_SUCCESS;
}

int PRISM_Prequest_free(PRISM_Request *request) {
	if (request == NULL || *request == NULL) return PRISM_SUCCESS;
	persistent_request_t *req = *request;
	if (req->active) return MPI_ERR_REQUEST;
	if (req->buffered && req->op_type == PERSISTENT_SEND && req->buffered_sent) {
		/* Keep the credit MR alive until all accepted messages are consumed. */
		do {
			int ret = mpiext_persistent_progress_request(req);
			if (ret != PRISM_SUCCESS) return ret;
		} while (req->flag_buffer < req->buffered_sent);
	}
	/* Retire control writes before releasing request context. */
	if (req->posted_ops || req->buffered_consumed) {
		int ret = mpiext_persistent_finalize_send_core(req);
		if (ret != PRISM_SUCCESS) return ret;
	}

	put_persistent_request(req->store_index);

	*request = NULL;

	return PRISM_SUCCESS;
}
