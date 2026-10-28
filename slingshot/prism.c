#include "prism.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <fi_cxi_ext.h>
#include <rdma/fi_trigger.h>

#define PRISM_ERROR -1
#define PRISM_SUCCESS 0

//#define DEBUG
#ifdef DEBUG
#    define OPT_PERSISTENT_INFO(fmt, ...)                                                  \
        fprintf(stdout, "[OPT PERSISTENT INFO] %s:%d:%s(): " fmt "\n", __FILE__, __LINE__, \
               __func__, ##__VA_ARGS__); fflush(stdout);
#else
#    define OPT_PERSISTENT_INFO(fmt, ...)
#endif

#define OPT_PERSISTENT_ERR(fmt, ...)                                                              \
    fprintf(stderr, "[OPT PERSISTENT ERROR] %s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, \
            ##__VA_ARGS__); fflush(stderr);

#define READY_TO_RECEIVE_FLAG      98
#define TRANSMISSION_COMPLETE_FLAG 99

#define PERSISTENT_LIKELY(x)   __builtin_expect(!!(x), 1)
#define PERSISTENT_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define MPIEXT_PERSISTENT_ADDR_KEY     "OPT_PERSISTENT_ADDR"
#define PERSISTENT_ALWAYS_INLINE       __attribute__((always_inline))

#define MPIEXT_PERSISTENT_DEFAULT_COMP 64
#define DEFAULT_REQUEST_STORE_SIZE UINT32_C(512)

/* CXI 2.3 client keys 0..99 select optimized MRs. A request needs one remote
 * MR (data or ready flag), or two data MRs for a buffered receiver.
 * Allocate the lowest free key so request-store rotation does not push
 * subsequent activations out of the optimized range. Keys >=100 provide
 * the standard-MR fallback when more than 100 requests coexist.
 */
#define PERSISTENT_MR_KEYS (2 * DEFAULT_REQUEST_STORE_SIZE)
#define PERSISTENT_SETUP_WORDS 10
#define PERSISTENT_SETUP_MAGIC UINT64_C(0x5052534358490005)
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

typedef struct mpiext_persistent_ctx {
	struct fi_info *fi;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_cq *cq;
	struct fid_cq *msg_cq;
	struct fid_av *av;
	struct fid_av *msg_av;
	struct fid_ep *ep;
	struct fid_ep *msg_ep;
	fi_addr_t *peer_addr;
	fi_addr_t *msg_peer_addr;
	uint64_t inject_size;
	int protocol; /* 0: RTR (default), 1: buffered, 2: triggered */
	struct persistent_request *pending_triggers;
	int error;
	int n_peers;
	int my_world_rank;
} mpiext_persistent_ctx_t;

typedef struct persistent_request {
	uint64_t remote_data_addr;
	uint64_t remote_flag_addr;
	uint64_t remote_data_mkey;
	uint64_t remote_flag_mkey;
	uint64_t posted_ops;
	uint64_t completed_ops;
	uint64_t expected_flag_ops;
	uint64_t buffered_sent, buffered_consumed, remote_second_mkey;
	volatile uint64_t credits[2];
	void *bounce_buffer;
	size_t remote_size;
	struct fid_mr *second_buffer_mr;
	struct fid_cntr *second_cntr, *trigger_cntr, *trigger_completion;
	int second_mr_key;
	bool buffered, active, trigger_pending, trigger_ready;
	int error;
	struct persistent_request *next_trigger;
	struct iovec trigger_iov;
	struct fi_rma_iov trigger_remote;
	struct fi_op_rma trigger_rma;
	struct fi_deferred_work trigger_work;
	size_t size;
	size_t my_rdma_info_size;
	struct fid_mr *data_buffer_mr;
	struct fid_mr *flag_buffer_mr;
        struct fid_cntr * data_access_cntr;
	void *data_buffer;
	void *my_rdma_info_buffer;
	uint32_t remote_store_index;
	volatile int flag_buffer;
	int tag;
	int peer_rank;
	fi_addr_t peer_addr;
	int store_index;
	int local_mr_key;
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

static persistent_request_t *request_store = NULL;
static request_element_t *free_requests_head = NULL;
static request_element_t *free_requests_tail = NULL;
static request_element_t *request_element_mpool = NULL;

static int mpiext_persistent_alloc_mr_key(void)
{
    /* Like other domain operations, serialized by the caller (FI_THREAD_DOMAIN). */
    for (uint32_t key = 0; key < PERSISTENT_MR_KEYS; ++key) {
        if (!remote_mr_keys[key]) {
            remote_mr_keys[key] = true;
            return (int)key;
        }
    }
    return -FI_ENOKEY;
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_cleanup_request_ressources(persistent_request_t *req) {
	int ret = 0;
	bool remote_mr_closed = true;
	bool second_closed = true;
	if (req->data_buffer_mr) {
		ret = fi_close(&req->data_buffer_mr->fid);
		if (ret < 0) {
			OPT_PERSISTENT_ERR("data buffer close failed");
			if (req->op_type == PERSISTENT_RECV)
				remote_mr_closed = false;
		}
	}
	if (req->flag_buffer_mr) {
		ret = fi_close(&req->flag_buffer_mr->fid);
		if (ret < 0) {
			OPT_PERSISTENT_ERR("flag buffer close failed");
			if (req->op_type == PERSISTENT_SEND)
				remote_mr_closed = false;
		}
	}
	if (req->data_access_cntr) {
		ret = fi_close(&req->data_access_cntr->fid);
		if (ret < 0) {
			OPT_PERSISTENT_ERR("data cntr close failed");
		}
	}
	if (req->second_buffer_mr) {
		if (fi_close(&req->second_buffer_mr->fid)) second_closed = false;
		req->second_buffer_mr = NULL;
	}
	if (req->second_cntr) fi_close(&req->second_cntr->fid);
	if (req->trigger_cntr) fi_close(&req->trigger_cntr->fid);
	if (req->trigger_completion) {
		fi_close(&req->trigger_completion->fid);
	}
	req->second_cntr = req->trigger_cntr = req->trigger_completion = NULL;
	if (req->second_mr_key >= 0 && second_closed) {
		remote_mr_keys[req->second_mr_key] = false;
		req->second_mr_key = -1;
	}
	if (remote_mr_closed && second_closed) free(req->bounce_buffer);
	req->bounce_buffer = NULL;
	/* Never recycle a key if its remote MR could not be closed. */
	if (req->local_mr_key >= 0 && remote_mr_closed) {
		remote_mr_keys[req->local_mr_key] = false;
		req->local_mr_key = -1;
	}
	free(req->my_rdma_info_buffer);
	req->my_rdma_info_buffer = NULL;

	req->data_access_cntr = NULL;
	req->data_buffer_mr = NULL;
	req->flag_buffer_mr = NULL;
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
	    malloc(DEFAULT_REQUEST_STORE_SIZE * sizeof(persistent_request_t));
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
	return 0;
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
	return elem;
}

static inline persistent_request_t *get_persistent_request(void) {
	if (!free_requests_head) return NULL;
	request_element_t *elem = free_requests_head;
	free_requests_head = elem->next;
	persistent_request_t *req = &request_store[elem->index];

	req->req_state = REQ_ACTIVE;
	req->store_index = elem->index;

	request_element_mpool_put(elem);

	return req;
}

static inline void put_persistent_request(int index) {
	request_element_t *elem = request_element_mpool_get();

	elem->index = index;
	elem->next = NULL;

	mpiext_persistent_cleanup_request_ressources(&request_store[index]);
	mpiext_persistent_reset_request(&request_store[index]);

	request_store[index].req_state = REQ_FREE;

	free_requests_append(elem);
}

#ifdef DEBUG
static void mpiext_persistent_print_provider_info(struct fi_info *info)
{
    if (NULL != info) {
        fprintf(stderr, "PRISM rank %d provider info:\n", ctx.my_world_rank);
        fprintf(stderr, "   %s (%s)\n", info->fabric_attr->prov_name, info->domain_attr->name);

        fprintf(stderr, "  capabilities: %s\n", fi_tostr(&info->caps, FI_TYPE_CAPS));
        fprintf(stderr, "  mode: %s\n", fi_tostr(&info->mode, FI_TYPE_MODE));
        if (info->mode & FI_CONTEXT)
            fprintf(stderr, "FI_CONTEXT\n");
        if (info->mode & FI_CONTEXT2)
            fprintf(stderr, "FI_CONTEXT2\n");

        struct fi_domain_attr *domain_attr = info->domain_attr;
        fprintf(stderr, "Domain attributes:\n");
        fprintf(stderr, "  threading: %s\n", fi_tostr(&domain_attr->threading, FI_TYPE_THREADING));
        fprintf(stderr, "  data progress mode %s\n",
                fi_tostr(&domain_attr->data_progress, FI_TYPE_PROGRESS));
        fprintf(stderr, "  control progress mode %s\n",
                fi_tostr(&domain_attr->control_progress, FI_TYPE_PROGRESS));
        fprintf(stderr, "  MR mode %s\n", fi_tostr(&domain_attr->mr_mode, FI_TYPE_MR_MODE));

        struct fi_tx_attr *tx_attr = info->tx_attr;
        fprintf(stderr, "Tx attributes:\n");
        fprintf(stderr, "  tx iov_limit: %ld\n", tx_attr->iov_limit);
        fprintf(stderr, "  tx rma_iov_limit: %ld\n", tx_attr->rma_iov_limit);
        fprintf(stderr, "  tx size: %zu\n", tx_attr->size);
        fprintf(stderr, "  tx ordering: %s\n",
                fi_tostr(&tx_attr->msg_order, FI_TYPE_MSG_ORDER));
        fprintf(stderr, "  tx default flags: %s\n",
                fi_tostr(&tx_attr->op_flags, FI_TYPE_OP_FLAGS));

        struct fi_cxi_dom_ops *ops = NULL;
        size_t depth = 0;
        int ret = fi_open_ops(&ctx.domain->fid, FI_CXI_DOM_OPS_4, 0,
                              (void **)&ops, NULL);
        if (!ret)
            ret = ops && ops->get_dwq_depth ?
                ops->get_dwq_depth(&ctx.domain->fid, &depth) : -FI_ENOSYS;
        if (!ret)
            fprintf(stderr, "  triggered operation capacity (shared CXI service/NIC): %zu\n", depth);
        else
            fprintf(stderr, "  triggered operation capacity: unavailable (%s)\n", fi_strerror(-ret));
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
	request->second_mr_key = -1;
	request->buffered = ctx.protocol == 1;
	request->posted_ops = 0;
	request->completed_ops = 0;
	request->expected_flag_ops = 0;
	request->remote_data_addr = 0;
	request->remote_flag_addr = 0;
	request->remote_data_mkey = 0;
	request->remote_flag_mkey = 0;
	request->data_access_cntr = NULL;
	request->data_buffer_mr = NULL;
	request->flag_buffer_mr = NULL;
	request->local_mr_key = -1;
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

int PRISM_Init(void)
{
    int ret;
    struct fi_info *hints = NULL;
    struct fi_cq_attr cq_attr = {0};
    struct fi_av_attr av_attr = {0};
    char addr[64];
    uint64_t addr_len = 64;
    int my_rank, world_size;
    struct fi_info *rma_info = NULL;

    const char *domain_name = getenv("PRISM_CXI_DOMAIN");
    
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ctx.my_world_rank = my_rank;
    ctx.n_peers = world_size;
    const char *mode = getenv("PRISM_CXI_COMPLETION");
    ctx.protocol = !mode || !strcmp(mode, "ct") || !strcmp(mode, "auto") ? 0 :
        !strcmp(mode, "ct-buffered") ? 1 : !strcmp(mode, "ct-triggered") ? 2 : -1;
    int min_mode, max_mode;
    PMPI_Allreduce(&ctx.protocol, &min_mode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    PMPI_Allreduce(&ctx.protocol, &max_mode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (min_mode < 0 || min_mode != max_mode) return MPI_ERR_ARG;

    hints = fi_allocinfo();
    if (!hints)
        return -FI_ENOMEM;

    hints->fabric_attr->prov_name = strdup("cxi");
    if (domain_name && domain_name[0])
        hints->domain_attr->name = strdup(domain_name);
    if (!hints->fabric_attr->prov_name ||
        (domain_name && domain_name[0] && !hints->domain_attr->name)) {
        fi_freeinfo(hints);
        return -FI_ENOMEM;
    }

    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_RMA_EVENT;
    if (ctx.protocol == 2) hints->caps |= FI_TRIGGER;
    //hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
    hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_ENDPOINT;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->addr_format = FI_FORMAT_UNSPEC;

    ret = fi_getinfo(FI_VERSION(2, 3), NULL, NULL, 0, hints, &rma_info);
    fi_freeinfo(hints);
    if (ret) {
        OPT_PERSISTENT_ERR("CXI lookup for domain %s failed: %s",
                           domain_name && domain_name[0] ? domain_name : "(provider default)",
                           fi_strerror(-ret));
        return ret;
    }

    ret = fi_fabric(rma_info->fabric_attr, &ctx.fabric, NULL);
    if (ret)
        goto err_cxi_setup;

    ret = fi_domain(ctx.fabric, rma_info, &ctx.domain, NULL);
    if (ret)
        goto err_cxi_setup;

#ifdef DEBUG
    mpiext_persistent_print_provider_info(rma_info);
#endif

    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;

    ret = fi_cq_open(ctx.domain, &cq_attr, &ctx.cq, NULL);
    assert(ret == 0);

    av_attr.type = FI_AV_TABLE;
    av_attr.count = world_size;

    ret = fi_av_open(ctx.domain, &av_attr, &ctx.av, NULL);
    assert(ret == 0);

    ret = fi_endpoint(ctx.domain, rma_info, &ctx.ep, NULL);
    assert(ret == 0);
    ret = fi_ep_bind(ctx.ep, &ctx.cq->fid, FI_TRANSMIT | FI_SELECTIVE_COMPLETION);
    assert(ret == 0);

    ret = fi_ep_bind(ctx.ep, &ctx.av->fid, 0);
    assert(ret == 0);

    /* Enable the endpoint: from this point data-path calls are valid. */
    ret = fi_enable(ctx.ep);
    assert(ret == 0);

    /* Publish our ep address */

    ret = fi_getname(&ctx.ep->fid, &addr[0], &addr_len);
    assert(ret == 0);

    OPT_PERSISTENT_INFO("My ep addr: %lu", (uint64_t) addr);
    OPT_PERSISTENT_INFO("My ep addr_len: %lu", addr_len);

    ctx.peer_addr = calloc(ctx.n_peers, sizeof(fi_addr_t));
  
    char *tmp_addr_buffer = malloc(world_size * addr_len);

    PMPI_Allgather(addr,addr_len,MPI_BYTE,tmp_addr_buffer,addr_len,MPI_BYTE,MPI_COMM_WORLD);

    for(int i = 0; i < world_size; ++i){
	if(i == my_rank)
		continue;
	ret = fi_av_insert(ctx.av,tmp_addr_buffer + i * addr_len, 1, &ctx.peer_addr[i],0,NULL);
	if(ret < 0){
		OPT_PERSISTENT_ERR("fi_av_insert failed");
	}
    }
   
    free(tmp_addr_buffer);
    ctx.inject_size = rma_info->tx_attr->inject_size;
    if (ctx.inject_size < sizeof(uint64_t)) { ret = -FI_ENOSYS; goto err_cxi_setup; }
    fi_freeinfo(rma_info);

    ret = init_request_storage();
    if (ret) return ret;
    OPT_PERSISTENT_INFO("Done initializing PRISM");
    return PRISM_SUCCESS;

err_cxi_setup:
    OPT_PERSISTENT_ERR("CXI setup failed: %s", fi_strerror(-ret));
    if (ctx.ep) { fi_close(&ctx.ep->fid); ctx.ep = NULL; }
    if (ctx.av) { fi_close(&ctx.av->fid); ctx.av = NULL; }
    if (ctx.cq) { fi_close(&ctx.cq->fid); ctx.cq = NULL; }
    free(ctx.peer_addr);
    ctx.peer_addr = NULL;
    if (ctx.domain) {
        fi_close(&ctx.domain->fid);
        ctx.domain = NULL;
    }
    if (ctx.fabric) {
        fi_close(&ctx.fabric->fid);
        ctx.fabric = NULL;
    }
    fi_freeinfo(rma_info);
    return ret;
}

static inline int mpiext_persistent_register_memory(persistent_request_t *request)
{
    struct fi_cntr_attr attr = {0};
    const bool receiving = request->op_type == PERSISTENT_RECV;
    int ret;

    request->local_mr_key = mpiext_persistent_alloc_mr_key();
    if (request->local_mr_key < 0)
        return request->local_mr_key;

    size_t stride = request->size ? request->size : 1;
    unsigned slots = receiving && request->buffered ? 2 : 1;
    if (slots == 2) {
        if (stride > SIZE_MAX / 2) { ret = -FI_EINVAL; goto err_register; }
        if (posix_memalign(&request->bounce_buffer, 64, 2 * stride)) {
            ret = -FI_ENOMEM;
            goto err_register;
        }
        request->second_mr_key = mpiext_persistent_alloc_mr_key();
        if (request->second_mr_key < 0) {
            ret = request->second_mr_key;
            goto err_register;
        }
    }
    for (unsigned slot = 0; slot < slots; ++slot) {
        struct fid_mr **mr = slot ? &request->second_buffer_mr : &request->data_buffer_mr;
        struct fid_cntr **cntr = slot ? &request->second_cntr : &request->data_access_cntr;
        void *buffer = request->bounce_buffer ?
            (char *)request->bounce_buffer + slot * stride : request->data_buffer;
        uint64_t key = slot ? request->second_mr_key : request->local_mr_key;
        ret = fi_mr_reg(ctx.domain, buffer, stride,
                        receiving ? FI_REMOTE_WRITE : FI_WRITE,
                        0, receiving ? key : 0, receiving ? FI_RMA_EVENT : 0,
                        mr, NULL);
        if (ret) goto err_register;
        ret = fi_mr_bind(*mr, &ctx.ep->fid, 0);
        if (ret) goto err_register;
        if (receiving) {
            attr.events = FI_CNTR_EVENTS_COMP;
            attr.wait_obj = FI_WAIT_NONE;
            attr.flags = FI_CXI_CNTR_CACHED;
            ret = fi_cntr_open(ctx.domain, &attr, cntr, NULL);
            if (ret) goto err_register;
            ret = fi_mr_bind(*mr, &(*cntr)->fid, FI_REMOTE_WRITE);
            if (ret) goto err_register;
        }
        ret = fi_mr_enable(*mr);
        if (ret) goto err_register;
    }
    if (!receiving && ctx.protocol == 2) {
        attr.events = FI_CNTR_EVENTS_COMP;
        attr.wait_obj = FI_WAIT_NONE;
        attr.flags = 0;
        ret = fi_cntr_open(ctx.domain, &attr, &request->trigger_cntr, NULL);
        if (ret) goto err_register;
        ret = fi_cntr_open(ctx.domain, &attr, &request->trigger_completion, NULL);
        if (ret) goto err_register;
    }

    /* Only the sender's flag is remotely written. Keeping the receiver's
     * flag local avoids spending a second optimized key per receive.
     */
    ret = fi_mr_reg(ctx.domain,
                    request->buffered ? (void *)request->credits : (void *)&request->flag_buffer,
                    request->buffered ? sizeof(request->credits) : sizeof(request->flag_buffer),
                    receiving ? FI_WRITE : FI_REMOTE_WRITE,
                    0, receiving ? 0 : (uint64_t)request->local_mr_key,
                    0, &request->flag_buffer_mr, NULL);
    if (ret)
        goto err_register;

    ret = fi_mr_bind(request->flag_buffer_mr, &ctx.ep->fid, 0ULL);
    if (ret)
        goto err_register;

    ret = fi_mr_enable(request->flag_buffer_mr);
    if (ret)
        goto err_register;

    OPT_PERSISTENT_INFO("[setup] DATA MR key (local)    : 0x%lx",
                        fi_mr_key(request->data_buffer_mr));
    OPT_PERSISTENT_INFO("[setup] DATA MR vaddr          : %p", (void *) request->data_buffer);
    OPT_PERSISTENT_INFO("[setup] FLAG MR key (local)    : 0x%lx",
                        fi_mr_key(request->flag_buffer_mr));
    OPT_PERSISTENT_INFO("[setup] FLAG MR vaddr          : %p", (void *) &request->flag_buffer);
    return PRISM_SUCCESS;

err_register:
    OPT_PERSISTENT_ERR("Request MR setup failed: %s", fi_strerror(-ret));
    mpiext_persistent_cleanup_request_ressources(request);
    return ret;
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_pack_rdma_info(persistent_request_t *request)
{
    uint64_t *packed = request->my_rdma_info_buffer;
    packed[0] = (uint64_t) request->data_buffer;
    packed[1] = request->op_type == PERSISTENT_RECV ?
                fi_mr_key(request->data_buffer_mr) : 0;
    packed[2] = (uint64_t) &request->flag_buffer;
    packed[3] = request->op_type == PERSISTENT_SEND ?
                fi_mr_key(request->flag_buffer_mr) : 0;
    packed[4] = (uint64_t)request->store_index;
    packed[5] = PERSISTENT_SETUP_MAGIC;
    packed[6] = ctx.protocol;
    packed[7] = request->size;
    packed[8] = request->second_buffer_mr ? fi_mr_key(request->second_buffer_mr) : 0;
    packed[9] = request->op_type;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_unpack_rdma_info(persistent_request_t *request, void *payload)
{
    uint64_t *packed = payload;
    if (packed[5] != PERSISTENT_SETUP_MAGIC || packed[6] != (uint64_t)ctx.protocol ||
        packed[9] != (uint64_t)(request->op_type == PERSISTENT_SEND ? PERSISTENT_RECV : PERSISTENT_SEND))
        return MPI_ERR_ARG;
    request->remote_size = packed[7];
    if (request->op_type == PERSISTENT_SEND ? request->size > request->remote_size :
                                            request->remote_size > request->size)
        return MPI_ERR_TRUNCATE;
    request->remote_second_mkey = packed[8];
    request->remote_data_addr = packed[0];
    request->remote_data_mkey = packed[1];
    request->remote_flag_addr = packed[2];
    request->remote_flag_mkey = packed[3];
    request->remote_store_index = (uint32_t)packed[4];
    return PRISM_SUCCESS;
}

static void mpiext_persistent_cleanup_module(void)
{
    if (ctx.ep)
        fi_close(&ctx.ep->fid);
    if (ctx.av)
        fi_close(&ctx.av->fid);
    if (ctx.cq)
        fi_close(&ctx.cq->fid);
    if (ctx.domain)
        fi_close(&ctx.domain->fid);
    if (ctx.fabric)
        fi_close(&ctx.fabric->fid);
    free(ctx.peer_addr);
    memset(&ctx, 0, sizeof(ctx));
}

int PRISM_Finalize(void)
{
    if (request_store) {
        for (unsigned i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i)
            if (request_store[i].req_state == REQ_ACTIVE && request_store[i].active)
                return MPI_ERR_REQUEST;
        for (unsigned i = 0; i < DEFAULT_REQUEST_STORE_SIZE; ++i) {
            if (request_store[i].req_state == REQ_FREE) continue;
            PRISM_Request request = &request_store[i];
            int ret = PRISM_Prequest_free(&request);
            if (ret) return ret;
        }
        cleanup_request_storage();
    }
    mpiext_persistent_cleanup_module();
    return PRISM_SUCCESS;
}

static int mpiext_persistent_init_request(persistent_request_t *request)
{
    int world_peer = request->peer_rank;
    if (request->req_comm != MPI_COMM_WORLD) {
        MPI_Group local, world;
        PMPI_Comm_group(request->req_comm, &local);
        PMPI_Comm_group(MPI_COMM_WORLD, &world);
        PMPI_Group_translate_ranks(local, 1, &request->peer_rank, world, &world_peer);
        PMPI_Group_free(&local);
        PMPI_Group_free(&world);
    }
    if (world_peer < 0 || world_peer >= ctx.n_peers) return MPI_ERR_RANK;
    request->peer_addr = ctx.peer_addr[world_peer];
    int ret = mpiext_persistent_register_memory(request);
    if (ret) return ret;
    request->my_rdma_info_size = PERSISTENT_SETUP_WORDS * sizeof(uint64_t);

    request->my_rdma_info_buffer = malloc(request->my_rdma_info_size);
    if (!request->my_rdma_info_buffer) return -FI_ENOMEM;

    mpiext_persistent_pack_rdma_info(request);

    // We assume eager protocol usage here
    ret = PMPI_Send(request->my_rdma_info_buffer, request->my_rdma_info_size, MPI_BYTE,
              request->peer_rank, request->tag, request->req_comm);
    return ret;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_progress(int count, struct fi_cq_entry cqes[])
{
    struct fi_cq_err_entry err = {0};
    if (ctx.error) return ctx.error;
    ssize_t rc = fi_cq_read(ctx.cq, cqes, count);
    if (rc > 0) {
        for (ssize_t i = 0; i < rc; ++i) {
            persistent_request_t *request = cqes[i].op_context;
            request->completed_ops++;
        }
    } else if (rc == -FI_EAVAIL) {
        fi_cq_readerr(ctx.cq, &err, 0);
        OPT_PERSISTENT_ERR("CQ: %s", fi_cq_strerror(ctx.cq, err.prov_errno, err.err_data, NULL, 0));
        return ctx.error = (int)rc;
    } else if (rc < 0 && rc != -FI_EAGAIN) {
        return ctx.error = (int)rc;
    }
    /* Poll every queued sender, including when waiting on a receive. These
     * trigger counters are deliberately not bound to the flag MRs. */
    persistent_request_t **link = &ctx.pending_triggers;
    while (*link) {
        persistent_request_t *request = *link;
        if (!request->trigger_ready) {
            if (request->flag_buffer != READY_TO_RECEIVE_FLAG) {
                link = &request->next_trigger;
                continue;
            }
            request->flag_buffer = 0;
            request->trigger_ready = true;
        }
        int ret = fi_cntr_set(request->trigger_cntr, request->expected_flag_ops);
        if (ret == -FI_EAGAIN) {
            link = &request->next_trigger;
            continue;
        }
        if (ret) return ctx.error = request->error = ret;
        *link = request->next_trigger;
        request->next_trigger = NULL;
    }
    return PRISM_SUCCESS;
}

/* Retain all deferred-work descriptors until the transfer completes. */
static int mpiext_persistent_start_triggered_transfer(persistent_request_t *request)
{
    if (request->expected_flag_ops == FI_CXI_CNTR_SUCCESS_MAX) return -FI_EOVERFLOW;
    request->trigger_iov = (struct iovec){request->data_buffer, request->size};
    request->trigger_remote = (struct fi_rma_iov){
        .addr = 0, .len = request->size, .key = request->remote_data_mkey,
    };
    request->trigger_rma = (struct fi_op_rma){
        .ep = ctx.ep,
        .msg = {.msg_iov = &request->trigger_iov, .iov_count = 1,
                .addr = request->peer_addr,
                .rma_iov = &request->trigger_remote, .rma_iov_count = 1,
                .context = request},
    };
    request->trigger_work = (struct fi_deferred_work){
        .threshold = request->expected_flag_ops + 1,
        .triggering_cntr = request->trigger_cntr,
        .completion_cntr = request->trigger_completion,
        .op_type = FI_OP_WRITE, .op.rma = &request->trigger_rma,
    };
    int ret;
    do {
        ret = fi_control(&ctx.domain->fid, FI_QUEUE_WORK, &request->trigger_work);
        if (ret == -FI_EAGAIN) {
            int progress = mpiext_persistent_progress(0, NULL);
            if (progress) return progress;
        }
    } while (ret == -FI_EAGAIN);
    if (ret) return ret;
    request->expected_flag_ops++;
    request->trigger_pending = true;
    request->trigger_ready = false;
    request->next_trigger = ctx.pending_triggers;
    ctx.pending_triggers = request;
    return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_data_transfer(persistent_request_t *request)
{
    ssize_t ret = -1;
    void *desc = fi_mr_desc(request->data_buffer_mr);
    uint64_t flags = 0;

    unsigned slot = request->buffered_sent % 2;
    if (request->buffered) {
        uint64_t needed = request->buffered_sent / 2;
        while (request->credits[slot] < needed) {
            int progress = mpiext_persistent_progress(0, NULL);
            if (progress) return progress;
        }
    }
    struct fi_rma_iov rma_iov = {
        .addr = 0,
        .len = request->size,
        .key = request->buffered && slot ? request->remote_second_mkey : request->remote_data_mkey,
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
        .context = (void *) request,
        .data = 0,
    };

    flags = 0;

    if (request->size <= ctx.inject_size) {
        flags = FI_INJECT;
    } else {
        /* CXI's default transmit completion permits source-buffer reuse.
         * Receiver visibility is established by its remote-write counter.
         */
        flags = FI_COMPLETION;
    }

    do {
        ret = fi_writemsg(ctx.ep, &rma_msg, flags);
        if (ret == -FI_EAGAIN) {
	    int progress = mpiext_persistent_progress(0, NULL);
	    if (progress) return progress;
        } else if (ret < 0) {
            OPT_PERSISTENT_ERR("fi_writemsg data failed with %s", fi_strerror(-ret));
            return ret;
        }
    } while (ret);

    if (flags & FI_COMPLETION) request->posted_ops++;
    if (request->buffered) request->buffered_sent++;
    return 0;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_flag_transfer(persistent_request_t *request)
{
    ssize_t ret = -1;
    unsigned slot = request->buffered_consumed % 2;
    uint64_t credit = request->buffered_consumed / 2 + 1;
    const void *source = request->buffered ? (const void *)&credit : (const void *)&request->flag_buffer;
    size_t length = request->buffered ? sizeof(credit) : sizeof(request->flag_buffer);
    uint64_t offset = request->buffered ? slot * sizeof(credit) : 0;
    do {
        ret = fi_inject_write(ctx.ep, source, length,
                              request->peer_addr, offset,
                              request->remote_flag_mkey);
        if (ret == -FI_EAGAIN) {
	    int progress = mpiext_persistent_progress(0, NULL);
	    if (progress) return progress;
        } else if (ret < 0) {
            OPT_PERSISTENT_ERR("fi_writemsg data failed with %s", fi_strerror(-ret));
            return ret;
        }
    } while (ret);

    return 0;
}

static inline int mpiext_persistent_start_send_core_no_sync(persistent_request_t *request)
{
    return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_start_send_core(persistent_request_t *request)
{
    if (request->buffered) return mpiext_persistent_start_data_transfer(request);
    if (ctx.protocol == 2) return mpiext_persistent_start_triggered_transfer(request);
    while (request->flag_buffer != READY_TO_RECEIVE_FLAG) {
        int ret = mpiext_persistent_progress(0, NULL);
        if (ret) return ret;
    }
    /* Consume this activation's RTR before issuing data. */
    request->flag_buffer = 0;
    return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_finalize_send_core(persistent_request_t *request)
{
    struct fi_cq_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];

    if (request->trigger_pending) {
        do {
            int ret = mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
            if (ret) return ret;
            if (fi_cntr_readerr(request->trigger_cntr) ||
                fi_cntr_readerr(request->trigger_completion)) return -FI_EIO;
        } while (fi_cntr_read(request->trigger_completion) < request->expected_flag_ops);
        request->trigger_pending = false;
    }
    while (request->posted_ops != request->completed_ops) {
        int ret = mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
        if (ret) return ret;
    }

    OPT_PERSISTENT_INFO("SEND OPERATION FINALIZED");
    return PRISM_SUCCESS;
}

static inline int mpiext_persistent_start_recv_core_no_sync(persistent_request_t *request)
{
    request->posted_ops++;
    return 0;
}

static inline int mpiext_persistent_start_recv_core(persistent_request_t *request)
{
    if (request->buffered) return mpiext_persistent_start_recv_core_no_sync(request);
    OPT_PERSISTENT_INFO("Start recv core");
    request->flag_buffer = READY_TO_RECEIVE_FLAG;
    int ret = mpiext_persistent_start_flag_transfer(request);
    if (!ret) request->posted_ops++;
    return ret;
}

PERSISTENT_ALWAYS_INLINE static inline uint64_t read_recv_completion(persistent_request_t *request)
{
    return fi_cntr_read(request->buffered && request->buffered_consumed % 2 ?
                        request->second_cntr : request->data_access_cntr);
}

static inline int mpiext_persistent_finalize_recv_core(persistent_request_t *request)
{
    unsigned slot = request->buffered_consumed % 2;
    uint64_t threshold = request->buffered ? request->buffered_consumed / 2 + 1 : request->posted_ops;
    struct fid_cntr *cntr = request->buffered && slot ? request->second_cntr : request->data_access_cntr;
    while (read_recv_completion(request) < threshold) {
        if (fi_cntr_readerr(cntr)) return -FI_EIO;
        if (ctx.protocol == 2) {
            int ret = mpiext_persistent_progress(0, NULL);
            if (ret) return ret;
        }
    }
    if (fi_cntr_readerr(cntr)) return -FI_EIO;
    if (request->buffered) {
        size_t stride = request->size ? request->size : 1;
        memcpy(request->data_buffer, (char *)request->bounce_buffer + slot * stride,
               request->remote_size);
        /* Reuse is permitted only after the application buffer has its copy. */
        int ret = mpiext_persistent_start_flag_transfer(request);
        if (ret) return ret;
        request->buffered_consumed++;
    }
    return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first_no_sync(persistent_request_t *request)
{
    int ret = 0;
OPT_PERSISTENT_INFO("req: %p peer_rank: %i",request,request->peer_rank);
    void *payload = malloc(request->my_rdma_info_size);
    assert(payload != NULL);

    ret = PMPI_Recv(payload, request->my_rdma_info_size, MPI_BYTE, request->peer_rank, request->tag,
               request->req_comm, MPI_STATUS_IGNORE);

    if (!ret) ret = mpiext_persistent_unpack_rdma_info(request, payload);

    free(payload);
    if (ret) return ret;

    request->init_state = READY;

    // Start operation after we finished init
    if (request->op_type == PERSISTENT_SEND) {
        OPT_PERSISTENT_INFO("ready send operation on %i", ctx.my_world_rank);
        return mpiext_persistent_start_send_core_no_sync(request);
    } else {
        OPT_PERSISTENT_INFO("ready recv operation on %i", ctx.my_world_rank);
        return mpiext_persistent_start_recv_core_no_sync(request);
    }
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first(persistent_request_t *request)
{
    int ret = 0;
    void *payload = malloc(request->my_rdma_info_size);
    assert(payload != NULL);

    ret = PMPI_Recv(payload, request->my_rdma_info_size, MPI_BYTE, request->peer_rank, request->tag,
               request->req_comm, MPI_STATUS_IGNORE);
    if (!ret) ret = mpiext_persistent_unpack_rdma_info(request, payload);

    free(payload);
    if (ret) return ret;

    request->init_state = READY;

    // Start operation after we finished init
    if (request->op_type == PERSISTENT_SEND) {
        OPT_PERSISTENT_INFO("ready send operation on %i", ctx.my_world_rank);
        return mpiext_persistent_start_send_core(request);
    } else {
        OPT_PERSISTENT_INFO("ready recv operation on %i", ctx.my_world_rank);
        return mpiext_persistent_start_recv_core(request);
    }
}

int PRISM_Psend_init(const void *buffer, int count, MPI_Datatype datatype,
		     int dst, int tag, MPI_Comm communicator,
		     PRISM_Request *request) {
	int size;
	if (!request) return MPI_ERR_ARG;
	*request = PRISM_REQUEST_NULL;
	if (!request_store || !ctx.ep) return MPI_ERR_OTHER;
	if (count < 0) return MPI_ERR_COUNT;
	int ret = PMPI_Type_size(datatype, &size);
	if (ret) return ret;
	if (size < 0 || (size && (size_t)count > SIZE_MAX / (size_t)size)) return MPI_ERR_COUNT;
	if (!buffer) return MPI_ERR_BUFFER;
	persistent_request_t *req = get_persistent_request();
	if (!req) return MPI_ERR_NO_MEM;

	req->req_comm = communicator;
	req->tag = tag;
	req->peer_rank = dst;
	req->data_buffer = (void *)buffer;
	req->op_type = PERSISTENT_SEND;

	req->size = (size_t)size * (size_t)count;


	OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p", (void *)buffer);
	ret = mpiext_persistent_init_request(req);
	if (ret) put_persistent_request(req->store_index);
	else *request = req;
	return ret;
}

int PRISM_Precv_init(void *buffer, int count, MPI_Datatype datatype, int src,
		     int tag, MPI_Comm communicator, PRISM_Request *request) {
	int size;
	if (!request) return MPI_ERR_ARG;
	*request = PRISM_REQUEST_NULL;
	if (!request_store || !ctx.ep) return MPI_ERR_OTHER;
	if (count < 0) return MPI_ERR_COUNT;
	int ret = PMPI_Type_size(datatype, &size);
	if (ret) return ret;
	if (size < 0 || (size && (size_t)count > SIZE_MAX / (size_t)size)) return MPI_ERR_COUNT;
	if (!buffer) return MPI_ERR_BUFFER;
	persistent_request_t *req = get_persistent_request();
	if (!req) return MPI_ERR_NO_MEM;


	req->req_comm = communicator;
	req->tag = tag;
	req->peer_rank = src;
	req->data_buffer = (void *)buffer;
	req->op_type = PERSISTENT_RECV;

	req->size = (size_t)size * (size_t)count;


	OPT_PERSISTENT_INFO("[PRECV_INIT]: data buffer %p", (void *)buffer);
	ret = mpiext_persistent_init_request(req);
	if (ret) put_persistent_request(req->store_index);
	else *request = req;
	return ret;
}

static int mpiext_persistent_start_send_internal_no_sync(persistent_request_t *request)
{
    if (PERSISTENT_LIKELY(request->init_state == READY))
        return mpiext_persistent_start_send_core_no_sync(request);
    else {
        return mpiext_persistent_finialize_init_first_no_sync(request);
    }
}

static int mpiext_persistent_start_recv_internal_no_sync(persistent_request_t *request)
{
    if (PERSISTENT_LIKELY(request->init_state == READY))
        return mpiext_persistent_start_recv_core_no_sync(request);
    else {
        return mpiext_persistent_finialize_init_first_no_sync(request);
    }
}

static int mpiext_persistent_start_send_internal(persistent_request_t *request)
{
    if (PERSISTENT_LIKELY(request->init_state == READY))
        return mpiext_persistent_start_send_core(request);
    else {
        return mpiext_persistent_finialize_init_first(request);
    }
}

static int mpiext_persistent_start_recv_internal(persistent_request_t *request)
{
    if (PERSISTENT_LIKELY(request->init_state == READY))
        return mpiext_persistent_start_recv_core(request);
    else {
        return mpiext_persistent_finialize_init_first(request);
    }
}

static int mpiext_persistent_start_internal(persistent_request_t *request)
{
    if (!request || request->active) return MPI_ERR_REQUEST;
    if (ctx.error || request->error) return ctx.error ? ctx.error : request->error;
    if (request->posted_ops == FI_CXI_CNTR_SUCCESS_MAX || request->buffered_sent == FI_CXI_CNTR_SUCCESS_MAX)
        return -FI_EOVERFLOW;
    request->active = true;
    int ret = request->op_type == PERSISTENT_SEND ?
        mpiext_persistent_start_send_internal(request) : mpiext_persistent_start_recv_internal(request);
    if (ret) request->error = ret;
    return ret;
}

static int mpiext_persistent_start_internal_no_sync(persistent_request_t *request)
{
    if (!request || request->active) return MPI_ERR_REQUEST;
    if (ctx.error || request->error) return ctx.error ? ctx.error : request->error;
    if (request->posted_ops == FI_CXI_CNTR_SUCCESS_MAX || request->buffered_sent == FI_CXI_CNTR_SUCCESS_MAX)
        return -FI_EOVERFLOW;
    request->active = true;
    int ret = request->op_type == PERSISTENT_SEND ?
        mpiext_persistent_start_send_internal_no_sync(request) : mpiext_persistent_start_recv_internal_no_sync(request);
    if (ret) request->error = ret;
    return ret;
}

int PRISM_Start_no_sync(PRISM_Request *request)
{
    return mpiext_persistent_start_internal_no_sync((persistent_request_t *) *request);
}

int PRISM_Startall_no_sync(int count, PRISM_Request request_array[])
{
    int ret = PRISM_ERROR;

    for (int i = 0; i < count; i++) {
        ret = mpiext_persistent_start_internal_no_sync(
            (persistent_request_t *) *(request_array + i));
        if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
            return PRISM_ERROR;
        }
    }
    return PRISM_SUCCESS;
}

int PRISM_Start(PRISM_Request *request)
{
	persistent_request_t *req = *request;
    return mpiext_persistent_start_internal(req);
}

int PRISM_Startall(int count, PRISM_Request request_array[])
{
    int ret = PRISM_ERROR;

    for (int i = 0; i < count; i++) {
        if (!request_array[i])
            continue;
        ret = mpiext_persistent_start_internal((persistent_request_t *) *(request_array + i));

        if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
            return PRISM_ERROR;
        }
    }
    return PRISM_SUCCESS;
}

static inline int mpiext_persistent_wait_internal(persistent_request_t *request)
{
    if (!request) return MPI_ERR_REQUEST;
    if (ctx.error || request->error) return ctx.error ? ctx.error : request->error;
    if (!request->active) return PRISM_SUCCESS;
    int ret = request->op_type == PERSISTENT_SEND ?
        mpiext_persistent_finalize_send_core(request) : mpiext_persistent_finalize_recv_core(request);
    if (ret) request->error = ret;
    else request->active = false;
    return ret;
}

int PRISM_Wait(PRISM_Request *request, MPI_Status *status)
{
    (void)status;
    return mpiext_persistent_wait_internal(*request);
}

int PRISM_Waitall(int count, PRISM_Request request_array[], MPI_Status status_array[])
{
    (void)status_array;
    int ret = PRISM_ERROR;

    for (int i = 0; i < count; i++) {
        if (!request_array[i])
            continue;
	persistent_request_t *req = *(&request_array[i]);
        ret = mpiext_persistent_wait_internal(req);
        if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
            return PRISM_ERROR;
        }
    }
    return PRISM_SUCCESS;
}

int PRISM_Prequest_free(PRISM_Request *request) {
	if (request == NULL || *request == NULL) return PRISM_SUCCESS;
	persistent_request_t *req = *request;
	if (req->active) return MPI_ERR_REQUEST;
	if (req->buffered && req->op_type == PERSISTENT_SEND) {
		/* Both slots must be consumed before their credit target disappears. */
		for (unsigned slot = 0; slot < 2; ++slot) {
			uint64_t needed = req->buffered_sent / 2 +
			    (slot == 0 && req->buffered_sent % 2);
			while (req->credits[slot] < needed) {
				int ret = mpiext_persistent_progress(0, NULL);
				if (ret) return ret;
			}
		}
	}
	put_persistent_request(req->store_index);
	*request = PRISM_REQUEST_NULL;

	return PRISM_SUCCESS;
}
