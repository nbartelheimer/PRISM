#include "uthash.h"
#include "prism.h"
#include <assert.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <stdio.h>

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
#define MPIEXT_PERSISTENT_MSG_ADDR_KEY "OPT_PERSISTENT_MSG_ADDR"
#define PERSISTENT_ALWAYS_INLINE       __attribute__((always_inline))

#define MPIEXT_PERSISTENT_DEFAULT_COMP 64

typedef enum mpiext_persistent_request_init_state {
    CHECK_FOR_REMOTE_INFO = 1,
    READY
} mpiext_persistent_request_init_state_t;

typedef enum mpiext_persistent_op_type {
    PERSISTENT_SEND = 1,
    PERSISTENT_RECV,
    OP_INVALID
} mpiext_persistent_op_type_t;

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
    int n_peers;
    int my_world_rank;
} mpiext_persistent_ctx_t;

typedef struct persistent_request {
    uint64_t remote_data_addr;
    uint64_t remote_flag_addr;
    uint64_t remote_data_mkey;
    uint64_t remote_flag_mkey;
    struct fid_cntr * data_access_cntr;
    struct fid_cntr * flag_access_cntr;
    uint64_t posted_ops;
    uint64_t completed_ops;
    uint64_t completed_msg_send;
    uint64_t expected_flag_ops;
    size_t size;
    size_t my_rdma_info_size;
    struct fid_mr *data_buffer_mr;
    struct fid_mr *flag_buffer_mr;
    void *data_buffer;
    void *my_rdma_info_buffer;
    uint32_t id;
    int flag_buffer;
    int tag;
    int peer_rank;
    int done_flag;
    int rtr_flag;
    mpiext_persistent_request_init_state_t init_state;
    mpiext_persistent_op_type_t op_type;
} persistent_request_t;

typedef struct request_entry {
    uint32_t id;
    persistent_request_t *request;
    UT_hash_handle hh;
} request_entry_t;

static mpiext_persistent_ctx_t ctx;
static request_entry_t *request_table = NULL;

PERSISTENT_ALWAYS_INLINE static inline uint32_t combine32(uint32_t a, uint32_t b)
{
    return a ^ (b + 0x9E3779B9u + (a << 6) + (a >> 2));
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_cleanup_request_ressources(persistent_request_t *req)
{
    fi_close(&req->data_buffer_mr->fid);
    fi_close(&req->flag_buffer_mr->fid);
    free(req->my_rdma_info_buffer);
}

static inline void mpiext_persistent_clear_request_table(void)
{
    request_entry_t *current_request, *tmp;
    HASH_ITER(hh, request_table, current_request, tmp)
    {
        assert(current_request != NULL);
        mpiext_persistent_cleanup_request_ressources(current_request->request);
        HASH_DEL(request_table, current_request);
        free(current_request);
    }
}

static inline void mpiext_persistent_remove_request_from_table(uint32_t id)
{
    request_entry_t *entry = NULL;

    HASH_FIND(hh, request_table, &id, sizeof(uint32_t), entry);
    assert(entry != NULL);
    mpiext_persistent_cleanup_request_ressources(entry->request);

    HASH_DEL(request_table, entry);

    free(entry);
}

static void mpiext_persistent_add_request_to_table(uint32_t id, persistent_request_t *request)
{
    request_entry_t *entry = malloc(sizeof(request_entry_t));
    assert(entry != NULL);
    entry->id = id;
    entry->request = request;
    OPT_PERSISTENT_INFO("Adding %s request with key %u to table on rank %i",
                        request->op_type == PERSISTENT_SEND ? "Send" : "Recv", id,
                        ctx.my_world_rank);

    HASH_ADD(hh, request_table, id, sizeof(uint32_t), entry);
}

static PERSISTENT_ALWAYS_INLINE persistent_request_t *
mpiext_persistent_get_request_from_table(uint32_t id)
{
    request_entry_t *entry = NULL;
    OPT_PERSISTENT_INFO("Checking for id %u in table", id);
    HASH_FIND(hh, request_table, &id, sizeof(uint32_t), entry);
    return entry->request;
}

#ifdef DEBUG
static void mpiext_persistent_print_provider_info(struct fi_info *info)
{
    if (NULL != info) {
        fprintf(stderr, "Provider info:\n");
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
    }
}
#endif

static inline void mpiext_persistent_reset_request(persistent_request_t *request)
{
    request->posted_ops = 0;
    request->completed_ops = 0;
    request->completed_msg_send = 0;
    request->expected_flag_ops = 0;
    request->remote_data_addr = 0;
    request->remote_flag_addr = 0;
    request->remote_data_mkey = 0;
    request->remote_flag_mkey = 0;
    request->data_buffer_mr = NULL;
    request->flag_buffer_mr = NULL;
    request->size = 0;
    request->id = 99;
    request->my_rdma_info_size = 0;
    request->data_buffer = NULL;
    request->my_rdma_info_buffer = NULL;
    request->flag_buffer = -1;
    request->tag = -1;
    request->peer_rank = -1;
    request->done_flag = TRANSMISSION_COMPLETE_FLAG;
    request->rtr_flag = READY_TO_RECEIVE_FLAG;
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
    
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ctx.my_world_rank = my_rank;
    ctx.n_peers = world_size;

    hints = fi_allocinfo();
    assert(hints != NULL);

    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_RMA | FI_TAGGED | FI_WRITE | FI_REMOTE_WRITE;
    hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
    hints->domain_attr->mr_mode = ~3;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->addr_format = FI_FORMAT_UNSPEC;

    ret = fi_getinfo(FI_VERSION(2, 2), NULL, NULL, 0, hints, &ctx.fi);
    assert(ret == 0);
    fi_freeinfo(hints);

#ifdef DEBUG
    if (my_rank == 0) {
        mpiext_persistent_print_provider_info(ctx.fi);
    }
#endif

    ret = fi_fabric(ctx.fi->fabric_attr, &ctx.fabric, NULL);
    assert(ret == 0);

    ret = fi_domain(ctx.fabric, ctx.fi, &ctx.domain, NULL);
    assert(ret == 0);

    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;

    ret = fi_cq_open(ctx.domain, &cq_attr, &ctx.cq, NULL);
    assert(ret == 0);

    av_attr.type = FI_AV_TABLE;
    av_attr.count = world_size;

    ret = fi_av_open(ctx.domain, &av_attr, &ctx.av, NULL);
    assert(ret == 0);

    ret = fi_endpoint(ctx.domain, ctx.fi, &ctx.ep, NULL);
    assert(ret == 0);
    ret = fi_ep_bind(ctx.ep, &ctx.cq->fid, FI_TRANSMIT | FI_SELECTIVE_COMPLETION | FI_RECV);
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
    ctx.msg_peer_addr = calloc(ctx.n_peers, sizeof(fi_addr_t));
    assert(ctx.peer_addr != NULL);

    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    cq_attr.wait_obj = FI_WAIT_NONE;

    ret = fi_cq_open(ctx.domain, &cq_attr, &ctx.msg_cq, NULL);
    assert(ret == 0);

    ret = fi_av_open(ctx.domain, &av_attr, &ctx.msg_av, NULL);
    assert(ret == 0);

    ret = fi_endpoint(ctx.domain, ctx.fi, &ctx.msg_ep, NULL);
    assert(ret == 0);
    ret = fi_ep_bind(ctx.msg_ep, &ctx.msg_cq->fid, FI_SEND | FI_RECV);
    assert(ret == 0);

    ret = fi_ep_bind(ctx.msg_ep, &ctx.msg_av->fid, 0);
    assert(ret == 0);

    /* Enable the endpoint: from this point data-path calls are valid. */
    ret = fi_enable(ctx.msg_ep);
    assert(ret == 0);

    char *tmp_addr_buffer = malloc(world_size * addr_len);
    char *tmp_msg_addr_buffer = malloc(world_size * addr_len);

    PMPI_Allgather(addr,addr_len,MPI_BYTE,tmp_addr_buffer,addr_len,MPI_BYTE,MPI_COMM_WORLD);

    for(int i = 0; i < world_size; ++i){
	if(i == my_rank)
		continue;
	ret = fi_av_insert(ctx.av,tmp_addr_buffer + i * addr_len, 1, &ctx.peer_addr[i],0,NULL);
	if(ret < 0){
		OPT_PERSISTENT_ERR("fi_av_insert failed");
	}
    }

    OPT_PERSISTENT_INFO("My msg_ep addr: %lu", (uint64_t) addr);
    OPT_PERSISTENT_INFO("My msg_ep addr_len: %lu", addr_len);

   /* Publish our ep address */
    memset(&addr[0], 0, 64);
    ret = fi_getname(&ctx.msg_ep->fid, &addr[0], &addr_len);
    assert(ret == 0);

    PMPI_Allgather(addr,addr_len,MPI_BYTE,tmp_msg_addr_buffer,addr_len,MPI_BYTE,MPI_COMM_WORLD);

    for(int i = 0; i < world_size; ++i){
	if(i == my_rank)
		continue;
	ret = fi_av_insert(ctx.msg_av,tmp_msg_addr_buffer + i * addr_len, 1, &ctx.msg_peer_addr[i],0,NULL);
	if(ret < 0){
		OPT_PERSISTENT_ERR("fi_av_insert failed");
	}
    }

    assert(ctx.msg_peer_addr != NULL);
    OPT_PERSISTENT_INFO("Done initializing PRISM");
    return PRISM_SUCCESS;
}

static inline int mpiext_persistent_register_memory(persistent_request_t *request)
{
    struct fi_cntr_attr attr = {0};
    int ret = fi_mr_reg(ctx.domain, request->data_buffer, request->size, FI_WRITE | FI_REMOTE_WRITE,
                        0, 0, 0, &request->data_buffer_mr, NULL);
    assert(ret == 0);
    if (ret < 0) {
        OPT_PERSISTENT_ERR("Data MR alloc failed");
        return ret;
    }

    attr.events = FI_CNTR_EVENTS_COMP;
    attr.wait_obj = FI_WAIT_UNSPEC;

    ret = fi_mr_bind(request->data_buffer_mr, &ctx.ep->fid, 0ULL);
    assert(ret == 0);

    ret = fi_cntr_open(ctx.domain,&attr,&request->data_access_cntr,NULL);
    assert(ret == 0);

    ret = fi_mr_bind(request->data_buffer_mr, &request->data_access_cntr->fid, FI_REMOTE_WRITE);
    assert(ret == 0);

    ret = fi_mr_enable(request->data_buffer_mr);
    assert(ret == 0);
	

    ret = fi_mr_reg(ctx.domain, &request->flag_buffer, sizeof(int), FI_WRITE | FI_REMOTE_WRITE, 0,
                    0, 0, &request->flag_buffer_mr, NULL);
    assert(ret == 0);
    if (ret < 0) {
        OPT_PERSISTENT_ERR("Flag MR alloc failed");
        return ret;
    }

    ret = fi_mr_bind(request->flag_buffer_mr, &ctx.ep->fid, 0ULL);
    assert(ret == 0);

    ret = fi_cntr_open(ctx.domain,&attr,&request->flag_access_cntr,NULL);
    assert(ret == 0);

    ret = fi_mr_bind(request->flag_buffer_mr, &request->flag_access_cntr->fid, FI_REMOTE_WRITE);
    assert(ret == 0);

    ret = fi_mr_enable(request->flag_buffer_mr);
    assert(ret == 0);

    OPT_PERSISTENT_INFO("[setup] DATA MR key (local)    : 0x%lx",
                        fi_mr_key(request->data_buffer_mr));
    OPT_PERSISTENT_INFO("[setup] DATA MR vaddr          : %p", (void *) request->data_buffer);
    OPT_PERSISTENT_INFO("[setup] FLAG MR key (local)    : 0x%lx",
                        fi_mr_key(request->flag_buffer_mr));
    OPT_PERSISTENT_INFO("[setup] FLAG MR vaddr          : %p", (void *) &request->flag_buffer);
    return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_pack_rdma_info(persistent_request_t *request)
{

    uint64_t *packed = request->my_rdma_info_buffer;
    packed[0] = (uint64_t) request->data_buffer;
    packed[1] = fi_mr_key(request->data_buffer_mr);
    packed[2] = (uint64_t) &request->flag_buffer;
    packed[3] = fi_mr_key(request->flag_buffer_mr);
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_unpack_rdma_info(persistent_request_t *request, void *payload)
{
	OPT_PERSISTENT_INFO("Unpacking req from peer: %i",request->peer_rank);
if(!payload)
OPT_PERSISTENT_INFO("Invalid payload");
    uint64_t *packed = payload;
    request->remote_data_addr = packed[0];
    request->remote_data_mkey = packed[1];
    request->remote_flag_addr = packed[2];
    request->remote_flag_mkey = packed[3];
}

static void mpiext_persistent_cleanup_module(void)
{
    if (ctx.av)
        fi_close(&ctx.av->fid);
    if (ctx.cq)
        fi_close(&ctx.cq->fid);
    if (ctx.ep)
        fi_close(&ctx.ep->fid);
    if (ctx.msg_av)
        fi_close(&ctx.msg_av->fid);
    if (ctx.msg_cq)
        fi_close(&ctx.msg_cq->fid);
    // if (ctx.msg_ep)
    //    fi_close(&ctx.msg_ep->fid);
    if (ctx.domain)
        fi_close(&ctx.domain->fid);
    if (ctx.fabric)
        fi_close(&ctx.fabric->fid);
    if (ctx.fi)
        fi_freeinfo(ctx.fi);
}

int PRISM_Finalize(void)
{
    mpiext_persistent_clear_request_table();
    mpiext_persistent_cleanup_module();
    return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int mpiext_persistent_msg_progress(void)
{
    struct fi_cq_tagged_entry cqes[MPIEXT_PERSISTENT_DEFAULT_COMP];
    struct fi_cq_err_entry err;
    persistent_request_t *request = NULL;
    ssize_t rc;

    rc = fi_cq_read(ctx.msg_cq, cqes, MPIEXT_PERSISTENT_DEFAULT_COMP);

    if (rc > 0) {
        for (ssize_t i = 0; i < rc; ++i) {
            if (cqes[i].flags & FI_RECV) {
            request = cqes[i].op_context;
                mpiext_persistent_unpack_rdma_info(request, request->my_rdma_info_buffer);
                request->init_state = READY;
            } else if (cqes[i].flags & FI_SEND) {
            request = cqes[i].op_context;
                request->completed_msg_send++;
            }
        }
    }

    if (PERSISTENT_UNLIKELY(rc == -FI_EAVAIL)) {
        fi_cq_readerr(ctx.cq, &err, 0);
        OPT_PERSISTENT_ERR("CQ: %s", fi_cq_strerror(ctx.cq, err.prov_errno, err.err_data, NULL, 0));
        return rc;
    }

    return 0;
}

static int mpiext_persistent_init_request(persistent_request_t *request)
{
    int ret = 0;
    mpiext_persistent_register_memory(request);
    request->my_rdma_info_size = 4 * sizeof(uint64_t);

    request->my_rdma_info_buffer = malloc(request->my_rdma_info_size);
    assert(request->my_rdma_info_buffer != NULL);

    mpiext_persistent_pack_rdma_info(request);

    // We assume eager protocol usage here
    // PMPI_Send(request->my_rdma_info_buffer, request->my_rdma_info_size, MPI_BYTE,
    //          request->peer_rank, request->tag, MPI_COMM_WORLD);
OPT_PERSISTENT_INFO("fi_tsend");
    do {
        ret = fi_tsend(ctx.msg_ep, request->my_rdma_info_buffer, request->my_rdma_info_size, NULL,
                       ctx.msg_peer_addr[request->peer_rank], request->id, request);
        if (ret == -FI_EAGAIN) {
            mpiext_persistent_msg_progress();
        } else if (ret < 0 && ret != FI_EAGAIN) {
            OPT_PERSISTENT_ERR("fi_tsend failed with %s", fi_strerror(-ret));
            return ret;
        }
    } while (ret);

    // Block until send completion
    while (request->completed_msg_send != 1) {
        mpiext_persistent_msg_progress();
    }

    return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_progress(int count, struct fi_cq_data_entry cqes[])
{
    struct fi_cq_err_entry err;
    persistent_request_t *request = NULL;
    ssize_t rc;

    rc = fi_cq_read(ctx.cq, cqes, count);

    if (rc > 0) {
        for (ssize_t i = 0; i < rc; ++i) {
            if (cqes[i].flags & FI_WRITE) {
OPT_PERSISTENT_INFO("finised loacal write");
                request = cqes[i].op_context;
                request->completed_ops++;
            }
        }
    }

    if (PERSISTENT_UNLIKELY(rc == -FI_EAVAIL)) {
        fi_cq_readerr(ctx.cq, &err, 0);
        OPT_PERSISTENT_ERR("CQ: %s", fi_cq_strerror(ctx.cq, err.prov_errno, err.err_data, NULL, 0));
        return rc;
    }

    return 0;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_data_transfer(persistent_request_t *request)
{
    ssize_t ret = -1;
    void *desc = fi_mr_desc(request->data_buffer_mr);
    uint64_t flags = 0;
    struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];

    struct fi_rma_iov rma_iov = {
        .addr = 0,
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
        .addr = ctx.peer_addr[request->peer_rank],
        .rma_iov = &rma_iov,
        .rma_iov_count = 1,
        .context = (void *) request,
        .data = 0,
    };

    flags = 0;

    if (request->size <= ctx.fi->tx_attr->inject_size) {
        flags |= FI_INJECT;
    } else {
        flags |= FI_COMPLETION | FI_INJECT_COMPLETE;
        request->posted_ops++;
    }

    do {
        ret = fi_writemsg(ctx.ep, &rma_msg, flags);
        if (ret == -FI_EAGAIN) {
	    (void)fi_cq_read(ctx.cq,NULL,0);
        } else if (ret < 0 && ret != FI_EAGAIN) {
            OPT_PERSISTENT_ERR("fi_writemsg data failed with %s", fi_strerror(-ret));
            return ret;
        }
    } while (ret);

    return 0;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_flag_transfer(persistent_request_t *request)
{
    struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];
    ssize_t ret = -1;

    do {
        ret = fi_inject_write(ctx.ep, &request->flag_buffer, sizeof(int),
                              ctx.peer_addr[request->peer_rank], 0,
                              request->remote_flag_mkey);
        if (ret == -FI_EAGAIN) {
	    (void)fi_cq_read(ctx.cq,NULL,0);
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
	request->expected_flag_ops++;
	uint64_t cur;
	for(;;){
		cur = fi_cntr_read(request->flag_access_cntr);
		if(cur == request->expected_flag_ops)
			break;
	}
    return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_finalize_send_core(persistent_request_t *request)
{
    struct fi_cq_data_entry cqe[MPIEXT_PERSISTENT_DEFAULT_COMP];

    while (request->posted_ops != request->completed_ops) {
        mpiext_persistent_progress(MPIEXT_PERSISTENT_DEFAULT_COMP, cqe);
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
OPT_PERSISTENT_INFO("Start recv core");
    request->flag_buffer = READY_TO_RECEIVE_FLAG;
    request->posted_ops++;
    return mpiext_persistent_start_flag_transfer(request);
}

static inline int mpiext_persistent_finalize_recv_core(persistent_request_t *request)
{
    uint64_t cur;
    for(;;){
		cur = fi_cntr_read(request->data_access_cntr);
		if(cur == request->posted_ops)
			break;	
    }
    OPT_PERSISTENT_INFO("Finalized Recv operation");
    return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first_no_sync(persistent_request_t *request)
{
    int ret = 0;
OPT_PERSISTENT_INFO("req: %p peer_rank: %i",request,request->peer_rank);
    void *payload = malloc(request->my_rdma_info_size);
    assert(payload != NULL);


    // PMPI_Recv(payload, request->my_rdma_info_size, MPI_BYTE, request->peer_rank, request->tag,
    //           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // mpiext_persistent_unpack_rdma_info(request, payload);
    do {
        ret = fi_trecv(ctx.msg_ep, payload, request->my_rdma_info_size, NULL,
                       ctx.msg_peer_addr[request->peer_rank], request->id, 0, request);
        if (ret == -FI_EAGAIN) {
            mpiext_persistent_msg_progress();
        } else if (ret < 0 && ret != FI_EAGAIN) {
            OPT_PERSISTENT_ERR("fi_tsend failed with %s", fi_strerror(-ret));
            return ret;
        }
    } while (ret);

    while (request->init_state != READY) {
        mpiext_persistent_msg_progress();
    }

    // important: set ready state here
    assert(request->init_state == READY);

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

    OPT_PERSISTENT_INFO("Posting init recv");
    // PMPI_Recv(payload, request->my_rdma_info_size, MPI_BYTE, request->peer_rank, request->tag,
    //           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // mpiext_persistent_unpack_rdma_info(request, payload);
    memset(request->my_rdma_info_buffer,0,request->my_rdma_info_size);
    do {
        ret = fi_trecv(ctx.msg_ep, request->my_rdma_info_buffer, request->my_rdma_info_size, NULL,
                       ctx.msg_peer_addr[request->peer_rank], request->id, 0, request);
        if (ret == -FI_EAGAIN) {
            mpiext_persistent_msg_progress();
        } else if (ret < 0 && ret != FI_EAGAIN) {
            OPT_PERSISTENT_ERR("fi_tsend failed with %s", fi_strerror(-ret));
            return ret;
        }
    } while (ret);

    OPT_PERSISTENT_INFO("Now progressing");
    while (request->init_state != READY) {
        mpiext_persistent_msg_progress();
    }

    free(payload);

    // important: check ready state here
    assert(request->init_state == READY);

    // Start operation after we finished init
    if (request->op_type == PERSISTENT_SEND) {
        OPT_PERSISTENT_INFO("ready send operation on %i", ctx.my_world_rank);
        return mpiext_persistent_start_send_core(request);
    } else {
        OPT_PERSISTENT_INFO("ready recv operation on %i", ctx.my_world_rank);
        return mpiext_persistent_start_recv_core(request);
    }
}

int PRISM_Psend_init(const void *buffer, int count, MPI_Datatype datatype, int dst, int tag,
                    MPI_Comm communicator, PRISM_Request *request)
{
    int size;
    OPT_PERSISTENT_INFO("Called PSend_init");
    persistent_request_t *req = malloc(sizeof(persistent_request_t));

    assert(req != NULL);

    OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p", (void *) buffer);
    mpiext_persistent_reset_request(req);

    // check if av entry already exists
    req->id = combine32((uint32_t) tag, (uint32_t) ctx.my_world_rank);
    req->tag = tag;
    req->peer_rank = dst;
    req->data_buffer = buffer;
    req->op_type = PERSISTENT_SEND;

    PMPI_Type_size(datatype, &size);
    req->size = size * count;

    mpiext_persistent_add_request_to_table(req->id, req);

    *request = (void *) req;
    OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p request: %p", (void *) buffer,req);
    return mpiext_persistent_init_request(req);
}

int PRISM_Precv_init(void *buffer, int count, MPI_Datatype datatype, int src, int tag,
                    MPI_Comm communicator, PRISM_Request *request)
{
    int size;
    OPT_PERSISTENT_INFO("Called Precv_init");
    persistent_request_t *req = malloc(sizeof(persistent_request_t));

    assert(req != NULL);

    mpiext_persistent_reset_request(req);

    // check if av entry already exists
    req->id = combine32((uint32_t) tag, (uint32_t) src);
    req->tag = tag;
    req->peer_rank = src;
    req->data_buffer = buffer;
    req->op_type = PERSISTENT_RECV;

    PMPI_Type_size(datatype, &size);
    req->size = size * count;

    mpiext_persistent_add_request_to_table(req->id, req);

    *request = (void *) req;
    OPT_PERSISTENT_INFO("[PRECV_INIT]: data buffer %p request: %p", (void *) buffer,req);
    return mpiext_persistent_init_request(req);
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
OPT_PERSISTENT_INFO("Checking op type");
    if (request->op_type == PERSISTENT_SEND){
OPT_PERSISTENT_INFO("op type send");
        return mpiext_persistent_start_send_internal(request);
}
    else{
	OPT_PERSISTENT_INFO("op type recv");
        return mpiext_persistent_start_recv_internal(request);
}
}

static int mpiext_persistent_start_internal_no_sync(persistent_request_t *request)
{
    if (request->op_type == PERSISTENT_SEND)
        return mpiext_persistent_start_send_internal_no_sync(request);
    else
        return mpiext_persistent_start_recv_internal_no_sync(request);
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
	OPT_PERSISTENT_INFO("start req: %p",(void*)*request);
	persistent_request_t *req = *request;
	OPT_PERSISTENT_INFO("start req peer %i",req->peer_rank);
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

static int mpiext_persistent_wait_internal(persistent_request_t *request)
{
    if (request->op_type == PERSISTENT_SEND)
        return mpiext_persistent_finalize_send_core(request);
    else
        return mpiext_persistent_finalize_recv_core(request);
}

int PRISM_Wait(PRISM_Request *request, MPI_Status *status)
{
    return mpiext_persistent_wait_internal((persistent_request_t *) *request);
}

int PRISM_Waitall(int count, PRISM_Request request_array[], MPI_Status status_array[])
{
    int ret = PRISM_ERROR;

    for (int i = 0; i < count; i++) {
        if (!request_array[i])
            continue;
        ret = mpiext_persistent_wait_internal((persistent_request_t *) *(request_array + i));
        if (PERSISTENT_UNLIKELY(PRISM_SUCCESS != ret)) {
            return PRISM_ERROR;
        }
    }
    return PRISM_SUCCESS;
}

int PRISM_Prequest_free(PRISM_Request *request)
{
    if (request == NULL)
        return PRISM_SUCCESS;

    persistent_request_t *req = (persistent_request_t *) *request;

    mpiext_persistent_remove_request_from_table(req->id);

    free(*request);
    *request = NULL;

    return PRISM_SUCCESS;
}
