#include "prism.h"

#include <assert.h>
#include <limits.h>
#include <portals4.h>
#include <stdio.h>

#include "uthash.h"

#define PRISM_ERROR -1
#define PRISM_SUCCESS 0

//#define DEBUG
#ifdef DEBUG
#define OPT_PERSISTENT_INFO(fmt, ...)                                  \
	fprintf(stdout, "[OPT PERSISTENT INFO] %s:%d:%s(): " fmt "\n", \
		__FILE__, __LINE__, __func__, ##__VA_ARGS__);          \
	fflush(stdout);
#else
#define OPT_PERSISTENT_INFO(fmt, ...)
#endif

#define OPT_PERSISTENT_ERR(fmt, ...)                                    \
	fprintf(stderr, "[OPT PERSISTENT ERROR] %s:%d:%s(): " fmt "\n", \
		__FILE__, __LINE__, __func__, ##__VA_ARGS__);           \
	fflush(stderr);

#define READY_TO_RECEIVE_FLAG 98
#define TRANSMISSION_COMPLETE_FLAG 99

#define PERSISTENT_LIKELY(x) __builtin_expect(!!(x), 1)
#define PERSISTENT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PERSISTENT_ALWAYS_INLINE __attribute__((always_inline))
#define DEFAULT_EQ_SIZE 1024
#define PRISM_ACK_TYPE PTL_NO_ACK_REQ

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
	ptl_handle_ni_t ni_h;
	ptl_process_t *peer_procs;
	int n_peers;
	int my_world_rank;
} mpiext_persistent_ctx_t;

typedef struct persistent_request {
	uint64_t remote_data_addr;
	uint64_t remote_flag_addr;
	uint64_t expected_md_ops;
	uint64_t expected_le_ops;
	ptl_handle_ct_t le_ct_h;
	ptl_handle_ct_t md_ct_h;
	ptl_handle_eq_t pt_eq_h;
	ptl_handle_md_t md_h;
	ptl_handle_le_t le_h;
	ptl_pt_index_t pt_idx;
	ptl_pt_index_t peer_pt_idx;
	size_t size;
	void *data_buffer;
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

PERSISTENT_ALWAYS_INLINE static inline uint32_t combine32(uint32_t a,
							  uint32_t b) {
	return a ^ (b + 0x9E3779B9u + (a << 6) + (a >> 2));
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_cleanup_request_ressources(persistent_request_t *req) {
	if (!PtlHandleIsEqual(req->md_ct_h, PTL_INVALID_HANDLE)) {
		PtlCTFree(req->md_ct_h);
	}

	if (!PtlHandleIsEqual(req->le_ct_h, PTL_INVALID_HANDLE)) {
		PtlCTFree(req->le_ct_h);
	}

	if (!PtlHandleIsEqual(req->md_h, PTL_INVALID_HANDLE)) {
		PtlMDRelease(req->md_h);
	}

	if (!PtlHandleIsEqual(req->le_h, PTL_INVALID_HANDLE)) {
		PtlLEUnlink(req->le_h);
	}

	if (!PtlHandleIsEqual(req->pt_eq_h, PTL_INVALID_HANDLE)) {
		PtlEQFree(req->pt_eq_h);
	}

	if (req->pt_idx) {
		PtlPTFree(ctx.ni_h, req->pt_idx);
	}
}

static inline void mpiext_persistent_clear_request_table(void) {
	request_entry_t *current_request, *tmp;
	HASH_ITER(hh, request_table, current_request, tmp) {
		assert(current_request != NULL);
		mpiext_persistent_cleanup_request_ressources(
		    current_request->request);
		HASH_DEL(request_table, current_request);
		free(current_request);
	}
}

static inline void mpiext_persistent_remove_request_from_table(uint32_t id) {
	request_entry_t *entry = NULL;

	HASH_FIND(hh, request_table, &id, sizeof(uint32_t), entry);
	assert(entry != NULL);
	mpiext_persistent_cleanup_request_ressources(entry->request);

	HASH_DEL(request_table, entry);

	free(entry);
}

static void mpiext_persistent_add_request_to_table(
    uint32_t id, persistent_request_t *request) {
	request_entry_t *entry = malloc(sizeof(request_entry_t));
	assert(entry != NULL);
	entry->id = id;
	entry->request = request;
	OPT_PERSISTENT_INFO(
	    "Adding %s request with key %u to table on rank %i",
	    request->op_type == PERSISTENT_SEND ? "Send" : "Recv", id,
	    ctx.my_world_rank);

	HASH_ADD(hh, request_table, id, sizeof(uint32_t), entry);
}

#ifdef DEBUG
void print_ni_limits(const ptl_ni_limits_t *limits) {
	printf("ni_req_limits.max_entries             = %d\n",
	       limits->max_entries);
	printf("ni_req_limits.max_unexpected_headers  = %d\n",
	       limits->max_unexpected_headers);
	printf("ni_req_limits.max_mds                 = %d\n", limits->max_mds);
	printf("ni_req_limits.max_eqs                 = %d\n", limits->max_eqs);
	printf("ni_req_limits.max_cts                 = %d\n", limits->max_cts);
	printf("ni_req_limits.max_pt_index            = %d\n",
	       limits->max_pt_index);
	printf("ni_req_limits.max_iovecs              = %d\n",
	       limits->max_iovecs);
	printf("ni_req_limits.max_list_size           = %d\n",
	       limits->max_list_size);
	printf("ni_req_limits.max_triggered_ops       = %d\n",
	       limits->max_triggered_ops);
	printf("ni_req_limits.max_msg_size             = %ld\n",
	       limits->max_msg_size);
	printf("ni_req_limits.max_atomic_size          = %ld\n",
	       limits->max_atomic_size);
	printf("ni_req_limits.max_fetch_atomic_size    = %ld\n",
	       limits->max_fetch_atomic_size);
	printf("ni_req_limits.max_waw_ordered_size     = %ld\n",
	       limits->max_waw_ordered_size);
	printf("ni_req_limits.max_war_ordered_size     = %ld\n",
	       limits->max_war_ordered_size);
	printf("ni_req_limits.max_volatile_size        = %ld\n",
	       limits->max_volatile_size);
	printf("ni_req_limits.features                 = 0x%lx\n",
	       (unsigned long)limits->features);
}
#endif

static inline void mpiext_persistent_reset_request(
    persistent_request_t *request) {
	request->expected_md_ops = 0;
	request->expected_le_ops = 0;
	request->le_ct_h = PTL_INVALID_HANDLE;
	request->md_ct_h = PTL_INVALID_HANDLE;
	request->pt_eq_h = PTL_INVALID_HANDLE;
	request->remote_data_addr = 0;
	request->remote_flag_addr = 0;
	request->size = 0;
	request->id = 99;
	request->data_buffer = NULL;
	request->flag_buffer = -1;
	request->tag = -1;
	request->peer_rank = -1;
	request->done_flag = TRANSMISSION_COMPLETE_FLAG;
	request->rtr_flag = READY_TO_RECEIVE_FLAG;
	request->init_state = CHECK_FOR_REMOTE_INFO;
	request->op_type = OP_INVALID;
}

int PRISM_Init(void) {
	int ret;
	int my_rank, world_size;
	ptl_ni_limits_t ni_req_limits;
	ptl_ni_limits_t ni_limits;
	ptl_process_t my_phys_addr;

	PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	PMPI_Comm_size(MPI_COMM_WORLD, &world_size);

	ctx.my_world_rank = my_rank;
	ctx.n_peers = world_size;

	ret = PtlInit();
	assert(ret == PTL_OK);

	ni_req_limits.max_entries = INT_MAX;
	ni_req_limits.max_unexpected_headers = INT_MAX;
	ni_req_limits.max_mds = INT_MAX;
	ni_req_limits.max_eqs = INT_MAX;
	ni_req_limits.max_cts = INT_MAX;
	ni_req_limits.max_pt_index = INT_MAX;
	ni_req_limits.max_iovecs = 0;
	ni_req_limits.max_list_size = INT_MAX;
	ni_req_limits.max_triggered_ops = INT_MAX;
	ni_req_limits.max_msg_size = LONG_MAX;
	ni_req_limits.max_atomic_size = LONG_MAX;
	ni_req_limits.max_fetch_atomic_size = LONG_MAX;
	ni_req_limits.max_waw_ordered_size = LONG_MAX;
	ni_req_limits.max_war_ordered_size = LONG_MAX;
	ni_req_limits.max_volatile_size = LONG_MAX;
	ni_req_limits.features = PTL_TARGET_BIND_INACCESSIBLE;

	ret = PtlNIInit(PTL_IFACE_DEFAULT, PTL_NI_NO_MATCHING | PTL_NI_PHYSICAL,
			PTL_PID_ANY, &ni_req_limits, &ni_limits, &ctx.ni_h);

#ifdef DEBUG
	print_ni_limits(&ni_limits);
#endif

	ret = PtlGetPhysId(ctx.ni_h, &my_phys_addr);
	assert(ret == PTL_OK);
	//printf("my_nid: %lu my_pid: %lu size: %lu\n", my_phys_addr.phys.nid,
	 //      my_phys_addr.phys.pid, sizeof(my_phys_addr.phys.nid));

	ctx.peer_procs = malloc(ctx.n_peers * sizeof(ptl_process_t));

	PMPI_Allgather(&my_phys_addr, sizeof(my_phys_addr), MPI_BYTE,
		       ctx.peer_procs, sizeof(ptl_process_t), MPI_BYTE,
		       MPI_COMM_WORLD);

	//if (my_rank == 0) {
	//	printf("other_nid: %lu other_pid: %lu\n",
	//	       ctx.peer_procs[1].phys.nid, ctx.peer_procs[1].phys.pid);
	//}
	OPT_PERSISTENT_INFO("Done initializing PRISM");
	return PRISM_SUCCESS;
}

static inline int mpiext_persistent_register_memory(
    persistent_request_t *request) {
	int ret = 0;
	unsigned options = 0;
	ptl_pt_index_t req_idx = PTL_PT_ANY;

	request->pt_eq_h = PTL_INVALID_HANDLE;

	ret = PtlEQAlloc(ctx.ni_h, DEFAULT_EQ_SIZE, &request->pt_eq_h);
	assert(ret == PTL_OK);

	ret = PtlCTAlloc(ctx.ni_h, &request->le_ct_h);
	assert(ret == 0);

	ret = PtlCTAlloc(ctx.ni_h, &request->md_ct_h);
	assert(ret == 0);

	ret = PtlPTAlloc(ctx.ni_h, options, request->pt_eq_h, req_idx,
			 &request->pt_idx);
	assert(ret == PTL_OK);

	OPT_PERSISTENT_INFO("Allocated PT idx: %lu size: %lu", request->pt_idx,
			    sizeof(request->pt_idx));

	unsigned le_options =
	    PTL_LE_OP_PUT | PTL_LE_EVENT_SUCCESS_DISABLE | PTL_LE_EVENT_CT_COMM;

	ptl_le_t le = {
	    .start = NULL,
	    .length = PTL_SIZE_MAX,
	    .uid = PTL_UID_ANY,
	    .options = le_options,
	    .ct_handle = request->le_ct_h,
	};

	ret = PtlLEAppend(ctx.ni_h, request->pt_idx, &le, PTL_PRIORITY_LIST,
			  NULL, &request->le_h);
	assert(ret == 0);

	ptl_event_t event;
	ret = PtlEQWait(request->pt_eq_h, &event);
	assert(ret == 0);

	assert(event.type == PTL_EVENT_LINK);
	assert(event.ni_fail_type == PTL_NI_OK);

	unsigned md_options = PTL_MD_VOLATILE | PTL_MD_EVENT_CT_SEND;

	ptl_md_t md = {
	    .start = NULL,
	    .length = PTL_SIZE_MAX,
	    .options = md_options,
	    .eq_handle = PTL_EQ_NONE,
	    .ct_handle = request->md_ct_h,
	};

	ret = PtlMDBind(ctx.ni_h, &md, &request->md_h);
	assert(ret == PTL_OK);
}

static void mpiext_persistent_cleanup_module(void) {
	PtlNIFini(ctx.ni_h);
	PtlFini();
}

int PRISM_Finalize(void) {
	mpiext_persistent_clear_request_table();
	mpiext_persistent_cleanup_module();
	return PRISM_SUCCESS;
}

static int mpiext_persistent_init_request(persistent_request_t *request) {
	uint64_t size = sizeof(unsigned) + 2 * sizeof(uint64_t);
	char *buffer = malloc(size);

	mpiext_persistent_register_memory(request);

	unsigned *idx = (unsigned *)buffer;
	idx[0] = request->pt_idx;

	uint64_t *addr = (uint64_t *)(buffer + sizeof(unsigned));
	addr[0] = (uint64_t)request->data_buffer;
	addr[1] = (uint64_t)&request->flag_buffer;

	PMPI_Send(buffer, size, MPI_BYTE, request->peer_rank, request->tag,
		  MPI_COMM_WORLD);

	free(buffer);
	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_data_transfer(persistent_request_t *request) {
	size_t local_offset = (size_t)request->data_buffer;
	size_t remote_offset = request->remote_data_addr;

	int ret = PtlPut(request->md_h, local_offset, request->size,
			 PRISM_ACK_TYPE, ctx.peer_procs[request->peer_rank],
			 request->peer_pt_idx, 0, remote_offset, NULL, 0);
	assert(ret == PTL_OK);
	request->expected_md_ops++;

	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_flag_transfer(persistent_request_t *request) {
	ptl_ct_event_t event;
	size_t local_offset = (size_t)&request->flag_buffer;
	size_t remote_offset = request->remote_flag_addr;
	int ret = PtlPut(request->md_h, local_offset, sizeof(int),
			 PRISM_ACK_TYPE, ctx.peer_procs[request->peer_rank],
			 request->peer_pt_idx, 0, remote_offset, NULL, 0);
	assert(ret == PTL_OK);
	request->expected_md_ops++;
	ret = PtlCTWait(request->md_ct_h, request->expected_md_ops, &event);
	assert(ret == PTL_OK);
	assert(event.failure == 0);

	return PRISM_SUCCESS;
}

static inline int mpiext_persistent_start_send_core_no_sync(
    persistent_request_t *request) {
	return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_start_send_core(
    persistent_request_t *request) {
	ptl_ct_event_t event;
	request->expected_le_ops++;
	OPT_PERSISTENT_INFO("Waiting on %lu flag buffer ops",
			    request->expected_le_ops);
	int ret = PtlCTWait(request->le_ct_h, request->expected_le_ops, &event);
	assert(ret == PTL_OK);
	assert(event.failure == 0);

	OPT_PERSISTENT_INFO("Got %lu flag buffer ops", event.success);

	return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_finalize_send_core(
    persistent_request_t *request) {
	ptl_ct_event_t event;
	int ret = PtlCTWait(request->md_ct_h, request->expected_md_ops, &event);
	assert(ret == PTL_OK);
	OPT_PERSISTENT_INFO("SEND OPERATION FINALIZED");
	return PRISM_SUCCESS;
}

static inline int mpiext_persistent_start_recv_core_no_sync(
    persistent_request_t *request) {
	return 0;
}

static inline int mpiext_persistent_start_recv_core(
    persistent_request_t *request) {
	OPT_PERSISTENT_INFO("Start recv core");
	request->flag_buffer = READY_TO_RECEIVE_FLAG;
	return mpiext_persistent_start_flag_transfer(request);
}

static inline int mpiext_persistent_finalize_recv_core(
    persistent_request_t *request) {
	ptl_ct_event_t event;

	request->expected_le_ops++;
	int ret = PtlCTWait(request->le_ct_h, request->expected_le_ops, &event);
	assert(ret == PTL_OK);
	assert(event.failure == 0);

	OPT_PERSISTENT_INFO("Finalized Recv operation");
	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first_no_sync(persistent_request_t *request) {
	uint64_t size = sizeof(unsigned) + 2 * sizeof(uint64_t);
	char *buffer = malloc(size);

	PMPI_Recv(buffer, size, MPI_BYTE, request->peer_rank, request->tag,
		  MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	unsigned *idx = (unsigned *)buffer;
	request->peer_pt_idx = *idx;

	uint64_t *addr = (uint64_t *)(buffer + sizeof(unsigned));
	request->remote_data_addr = addr[0];
	request->remote_flag_addr = addr[1];

	free(buffer);

	request->init_state = READY;

	// Start operation after we finished init
	if (request->op_type == PERSISTENT_SEND) {
		OPT_PERSISTENT_INFO("ready send operation on %i",
				    ctx.my_world_rank);
		return mpiext_persistent_start_send_core_no_sync(request);
	} else {
		OPT_PERSISTENT_INFO("ready recv operation on %i",
				    ctx.my_world_rank);
		return mpiext_persistent_start_recv_core_no_sync(request);
	}
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_finialize_init_first(persistent_request_t *request) {
	uint64_t size = sizeof(unsigned) + 2 * sizeof(uint64_t);
	char *buffer = malloc(size);

	PMPI_Recv(buffer, size, MPI_BYTE, request->peer_rank, request->tag,
		  MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	unsigned *idx = (unsigned *)buffer;
	request->peer_pt_idx = *idx;

	uint64_t *addr = (uint64_t *)(buffer + sizeof(unsigned));
	request->remote_data_addr = addr[0];
	request->remote_flag_addr = addr[1];

	free(buffer);

	request->init_state = READY;

	// Start operation after we finished init
	if (request->op_type == PERSISTENT_SEND) {
		OPT_PERSISTENT_INFO("ready send operation on %i",
				    ctx.my_world_rank);
		return mpiext_persistent_start_send_core(request);
	} else {
		OPT_PERSISTENT_INFO("ready recv operation on %i",
				    ctx.my_world_rank);
		return mpiext_persistent_start_recv_core(request);
	}
}

int PRISM_Psend_init(const void *buffer, int count, MPI_Datatype datatype,
		     int dst, int tag, MPI_Comm communicator,
		     PRISM_Request *request) {
	int size;
	OPT_PERSISTENT_INFO("Called PSend_init");
	persistent_request_t *req = malloc(sizeof(persistent_request_t));

	assert(req != NULL);

	OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p", (void *)buffer);
	mpiext_persistent_reset_request(req);

	// check if av entry already exists
	req->id = combine32((uint32_t)tag, (uint32_t)ctx.my_world_rank);
	req->tag = tag;
	req->peer_rank = dst;
	req->data_buffer = buffer;
	req->op_type = PERSISTENT_SEND;

	PMPI_Type_size(datatype, &size);
	req->size = size * count;

	mpiext_persistent_add_request_to_table(req->id, req);

	*request = (void *)req;
	OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p request: %p",
			    (void *)buffer, req);
	return mpiext_persistent_init_request(req);
}

int PRISM_Precv_init(void *buffer, int count, MPI_Datatype datatype, int src,
		     int tag, MPI_Comm communicator, PRISM_Request *request) {
	int size;
	OPT_PERSISTENT_INFO("Called Precv_init");
	persistent_request_t *req = malloc(sizeof(persistent_request_t));

	assert(req != NULL);

	mpiext_persistent_reset_request(req);

	// check if av entry already exists
	req->id = combine32((uint32_t)tag, (uint32_t)src);
	req->tag = tag;
	req->peer_rank = src;
	req->data_buffer = buffer;
	req->op_type = PERSISTENT_RECV;

	PMPI_Type_size(datatype, &size);
	req->size = size * count;

	mpiext_persistent_add_request_to_table(req->id, req);

	*request = (void *)req;
	OPT_PERSISTENT_INFO("[PRECV_INIT]: data buffer %p request: %p",
			    (void *)buffer, req);
	return mpiext_persistent_init_request(req);
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
	OPT_PERSISTENT_INFO("Checking op type");
	if (request->op_type == PERSISTENT_SEND) {
		OPT_PERSISTENT_INFO("op type send");
		return mpiext_persistent_start_send_internal(request);
	} else {
		OPT_PERSISTENT_INFO("op type recv");
		return mpiext_persistent_start_recv_internal(request);
	}
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
	OPT_PERSISTENT_INFO("start req: %p", (void *)*request);
	persistent_request_t *req = *request;
	OPT_PERSISTENT_INFO("start req peer %i", req->peer_rank);
	return mpiext_persistent_start_internal(req);
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
	if (request == NULL) return PRISM_SUCCESS;

	persistent_request_t *req = (persistent_request_t *)*request;
	OPT_PERSISTENT_INFO("Delte request with id %u",req->id);

	mpiext_persistent_remove_request_from_table(req->id);

	free(*request);
	*request = NULL;

	return PRISM_SUCCESS;
}
