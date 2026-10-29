#include "prism.h"

#include <assert.h>
#include <limits.h>
#include <portals4.h>
#include <portals4_bxiext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uthash.h"

#define PRISM_ERROR -1
#define PRISM_SUCCESS 0

/* Enable verbose tracing explicitly; it must be off for benchmarking. */
#ifdef PRISM_DEBUG
#define DEBUG
#endif
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
	int buffered;
	int triggered;
	struct persistent_request *pending_sends;
} mpiext_persistent_ctx_t;

typedef struct persistent_request {
	uint64_t remote_data_addr;
	uint64_t remote_flag_addr;
	uint64_t expected_md_ops;
	uint64_t expected_le_ops;
	/* Buffered mode: sequence is the number of completed activations. */
	uint64_t sequence;
	uint64_t credits;
	void *bounce_buffer;
	size_t bounce_stride;
	ptl_handle_ct_t second_ct_h;
	ptl_handle_le_t second_le_h;
	ptl_pt_index_t second_pt_idx;
	ptl_pt_index_t peer_second_pt_idx;
	int active;
	int pending_send;
	int copied;
	struct persistent_request *next_pending;
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

static int mpiext_persistent_drain_credits(persistent_request_t *request);

PERSISTENT_ALWAYS_INLINE static inline uint32_t combine32(uint32_t a,
							  uint32_t b) {
	return a ^ (b + 0x9E3779B9u + (a << 6) + (a >> 2));
}

PERSISTENT_ALWAYS_INLINE static inline void
mpiext_persistent_cleanup_request_ressources(persistent_request_t *req) {
	if (!PtlHandleIsEqual(req->md_h, PTL_INVALID_HANDLE)) {
		PtlMDRelease(req->md_h);
	}

	if (!PtlHandleIsEqual(req->le_h, PTL_INVALID_HANDLE)) {
		PtlLEUnlink(req->le_h);
	}
	if (!PtlHandleIsEqual(req->second_le_h, PTL_INVALID_HANDLE)) {
		PtlLEUnlink(req->second_le_h);
	}

	if (!PtlHandleIsEqual(req->md_ct_h, PTL_INVALID_HANDLE)) {
		PtlCTFree(req->md_ct_h);
	}
	if (!PtlHandleIsEqual(req->le_ct_h, PTL_INVALID_HANDLE)) {
		PtlCTFree(req->le_ct_h);
	}
	if (!PtlHandleIsEqual(req->second_ct_h, PTL_INVALID_HANDLE)) {
		PtlCTFree(req->second_ct_h);
	}

	if (req->pt_idx != (ptl_pt_index_t)PTL_PT_ANY) {
		PtlPTFree(ctx.ni_h, req->pt_idx);
	}
	if (req->second_pt_idx != (ptl_pt_index_t)PTL_PT_ANY) {
		PtlPTFree(ctx.ni_h, req->second_pt_idx);
	}
	if (!PtlHandleIsEqual(req->pt_eq_h, PTL_INVALID_HANDLE)) {
		PtlEQFree(req->pt_eq_h);
	}
	free(req->bounce_buffer);
}

static inline void mpiext_persistent_clear_request_table(void) {
	request_entry_t *current_request, *tmp;
	HASH_ITER(hh, request_table, current_request, tmp) {
		assert(current_request != NULL);
		mpiext_persistent_cleanup_request_ressources(
		    current_request->request);
		HASH_DEL(request_table, current_request);
		free(current_request->request);
		free(current_request);
	}
}

static inline void mpiext_persistent_remove_request_from_table(persistent_request_t *request) {
	request_entry_t *entry = NULL;

	HASH_FIND_PTR(request_table, &request, entry);
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

	/* Different requests may have the same tag and peer. */
	HASH_ADD_PTR(request_table, request, entry);
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
	request->sequence = 0;
	request->credits = 0;
	request->bounce_buffer = NULL;
	request->bounce_stride = 0;
	request->second_ct_h = PTL_INVALID_HANDLE;
	request->second_le_h = PTL_INVALID_HANDLE;
	request->second_pt_idx = PTL_PT_ANY;
	request->peer_second_pt_idx = PTL_PT_ANY;
	request->active = 0;
	request->pending_send = 0;
	request->copied = 0;
	request->next_pending = NULL;
	request->md_h = PTL_INVALID_HANDLE;
	request->le_h = PTL_INVALID_HANDLE;
	request->pt_idx = PTL_PT_ANY;
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
	const char *completion = getenv("PRISM_PORTALS4_COMPLETION");
	ctx.buffered = completion && strcmp(completion, "ct-buffered") == 0;
	ctx.triggered = completion && strcmp(completion, "ct-triggered") == 0;
	ctx.pending_sends = NULL;
	int mode = ctx.triggered ? 2 : ctx.buffered ? 1 : 0, min_mode, max_mode;
	if (completion && strcmp(completion, "ct-buffered") != 0 &&
	    strcmp(completion, "ct-triggered") != 0 &&
	    strcmp(completion, "ct") != 0) mode = -1;
	PMPI_Allreduce(&mode, &min_mode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	PMPI_Allreduce(&mode, &max_mode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (min_mode < 0 || min_mode != max_mode) {
		OPT_PERSISTENT_ERR("PRISM_PORTALS4_COMPLETION must be ct, ct-buffered or ct-triggered and identical on all ranks");
		return PRISM_ERROR;
	}

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
	/* BXI treats INT_MAX as unspecified, leaving no usable trigger capacity. */
	ni_req_limits.max_triggered_ops = ctx.triggered ? 1024 : INT_MAX;
	ni_req_limits.max_msg_size = LONG_MAX;
	ni_req_limits.max_atomic_size = LONG_MAX;
	ni_req_limits.max_fetch_atomic_size = LONG_MAX;
	ni_req_limits.max_waw_ordered_size = LONG_MAX;
	ni_req_limits.max_war_ordered_size = LONG_MAX;
	ni_req_limits.max_volatile_size = LONG_MAX;
	ni_req_limits.features = PTL_TARGET_BIND_INACCESSIBLE;

	ret = PtlNIInit(PTL_IFACE_DEFAULT, PTL_NI_NO_MATCHING | PTL_NI_PHYSICAL,
			PTL_PID_ANY, &ni_req_limits, &ni_limits, &ctx.ni_h);
	assert(ret == PTL_OK);

	if (ctx.triggered) {
		int capacity = ni_limits.max_triggered_ops, minimum;
		PMPI_Allreduce(&capacity, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		if (minimum < 1) {
			OPT_PERSISTENT_ERR("Triggered puts unavailable on at least one rank");
			PtlNIFini(ctx.ni_h);
			PtlFini();
			return PRISM_ERROR;
		}
	}

#ifdef DEBUG
	print_ni_limits(&ni_limits);
#endif

	ret = PtlGetPhysId(ctx.ni_h, &my_phys_addr);
	assert(ret == PTL_OK);
	// printf("my_nid: %lu my_pid: %lu size: %lu\n", my_phys_addr.phys.nid,
	//       my_phys_addr.phys.pid, sizeof(my_phys_addr.phys.nid));

	ctx.peer_procs = malloc(ctx.n_peers * sizeof(ptl_process_t));

	PMPI_Allgather(&my_phys_addr, sizeof(my_phys_addr), MPI_BYTE,
		       ctx.peer_procs, sizeof(ptl_process_t), MPI_BYTE,
		       MPI_COMM_WORLD);

	// if (my_rank == 0) {
	//	printf("other_nid: %lu other_pid: %lu\n",
	//	       ctx.peer_procs[1].phys.nid, ctx.peer_procs[1].phys.pid);
	// }
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
	assert(ret == PTL_OK);
	ret = PtlCTAlloc(ctx.ni_h, &request->md_ct_h);
	assert(ret == PTL_OK);

	ret = PtlPTAlloc(ctx.ni_h, options, request->pt_eq_h, req_idx,
			 &request->pt_idx);
	assert(ret == PTL_OK);

	OPT_PERSISTENT_INFO("Allocated PT idx: %lu size: %lu", request->pt_idx,
			    sizeof(request->pt_idx));

	unsigned le_options = PTL_LE_OP_PUT | PTL_LE_EVENT_SUCCESS_DISABLE |
	                      PTL_LE_EVENT_CT_COMM;

	ptl_le_t le = {
	    .start = NULL,
	    .length = PTL_SIZE_MAX,
	    .uid = PTL_UID_ANY,
	    .options = le_options,
	    .ct_handle = request->le_ct_h,
	};
	if (request->bounce_buffer) {
		le.start = request->bounce_buffer;
		le.length = request->size;
	}

	ret = PtlLEAppend(ctx.ni_h, request->pt_idx, &le, PTL_PRIORITY_LIST,
			  request, &request->le_h);
	assert(ret == 0);

	ptl_event_t event;
	ret = PtlEQWait(request->pt_eq_h, &event);
	assert(ret == 0);

	assert(event.type == PTL_EVENT_LINK);
	assert(event.ni_fail_type == PTL_NI_OK);

	if (request->bounce_buffer) {
		/* Independent counters: slot 1 may finish before slot 0, especially
		 * for messages beyond the NI's ordered-write size. */
		ret = PtlCTAlloc(ctx.ni_h, &request->second_ct_h);
		assert(ret == PTL_OK);
		ret = PtlPTAlloc(ctx.ni_h, 0, request->pt_eq_h, PTL_PT_ANY,
		                 &request->second_pt_idx);
		assert(ret == PTL_OK);
		le.start = (char *)request->bounce_buffer + request->bounce_stride;
		le.ct_handle = request->second_ct_h;
		ret = PtlLEAppend(ctx.ni_h, request->second_pt_idx, &le,
		                  PTL_PRIORITY_LIST, request, &request->second_le_h);
		assert(ret == PTL_OK);
		ret = PtlEQWait(request->pt_eq_h, &event);
		assert(ret == PTL_OK && event.type == PTL_EVENT_LINK &&
		       event.ni_fail_type == PTL_NI_OK);
	}

	unsigned md_options = PTL_MD_VOLATILE | PTL_MD_EVENT_SUCCESS_DISABLE |
	                      PTL_MD_EVENT_SEND_DISABLE | PTL_MD_EVENT_CT_SEND;

	ptl_md_t md = {
	    .start = NULL,
	    .length = PTL_SIZE_MAX,
	    .options = md_options,
	    .eq_handle = PTL_EQ_NONE,
	    .ct_handle = request->md_ct_h,
	};

	ret = PtlMDBind(ctx.ni_h, &md, &request->md_h);
	assert(ret == PTL_OK);
	return PRISM_SUCCESS;
}

static void mpiext_persistent_cleanup_module(void) {
	PtlNIFini(ctx.ni_h);
	PtlFini();
	free(ctx.peer_procs);
}

int PRISM_Finalize(void) {
	if (ctx.buffered) {
		request_entry_t *entry, *tmp;
		HASH_ITER(hh, request_table, entry, tmp) {
			if (entry->request->active) return PRISM_ERROR;
		}
		HASH_ITER(hh, request_table, entry, tmp) {
			if (mpiext_persistent_drain_credits(entry->request) != PRISM_SUCCESS)
				return PRISM_ERROR;
		}
	}
	mpiext_persistent_clear_request_table();
	mpiext_persistent_cleanup_module();
	return PRISM_SUCCESS;
}

typedef struct persistent_remote_info {
	uint64_t data_addr;
	uint64_t flag_addr;
	uint64_t size;
	ptl_pt_index_t pt_idx;
	ptl_pt_index_t second_pt_idx;
	int op_type;
} persistent_remote_info_t;

static int mpiext_persistent_init_request(persistent_request_t *request) {
	if (ctx.buffered && request->op_type == PERSISTENT_RECV) {
		request->bounce_stride = request->size ? request->size : 1;
		if (request->bounce_stride > SIZE_MAX / 2) return PRISM_ERROR;
		request->bounce_buffer = malloc(2 * request->bounce_stride);
		if (!request->bounce_buffer) return PRISM_ERROR;
	}
	mpiext_persistent_register_memory(request);
	persistent_remote_info_t info = {
	    .data_addr = (uintptr_t)request->data_buffer,
	    .flag_addr = (uintptr_t)&request->flag_buffer,
	    .size = request->size,
	    .pt_idx = request->pt_idx,
	    .second_pt_idx = request->second_pt_idx,
	    .op_type = request->op_type,
	};
	return PMPI_Send(&info, sizeof(info), MPI_BYTE, request->peer_rank,
	                 request->tag, MPI_COMM_WORLD) == MPI_SUCCESS ?
	                 PRISM_SUCCESS : PRISM_ERROR;
}

static int mpiext_persistent_finish_setup(persistent_request_t *request) {
	persistent_remote_info_t info;
	int ret = PMPI_Recv(&info, sizeof(info), MPI_BYTE, request->peer_rank,
	                    request->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	if (ret != MPI_SUCCESS) return PRISM_ERROR;
	if (ctx.buffered && (info.size != request->size ||
	    info.op_type == request->op_type)) {
		OPT_PERSISTENT_ERR("Buffered setup requires matching send/receive sizes and directions");
		return PRISM_ERROR;
	}
	request->peer_pt_idx = info.pt_idx;
	request->peer_second_pt_idx = info.second_pt_idx;
	request->remote_data_addr = info.data_addr;
	request->remote_flag_addr = info.flag_addr;
	request->init_state = READY;
	return PRISM_SUCCESS;
}

/* The first two slots are granted by setup. Each subsequent send requires
 * one more returned credit. Credits count consumed messages, not posted recvs.
 * Never bypass this check, including through the no-sync API. */
static int mpiext_persistent_buffered_try_send(persistent_request_t *request,
                                             int *posted) {
	*posted = 0;
	uint64_t needed = request->sequence > 1 ? request->sequence - 1 : 0;
	if (request->credits < needed) {
		ptl_ct_event_t event;
		int ret = PtlCTGet(request->le_ct_h, &event);
		if (ret != PTL_OK || event.failure) {
			OPT_PERSISTENT_ERR("Credit counter failed");
			return PRISM_ERROR;
		}
		request->credits = event.success;
		if (request->credits < needed) return PRISM_SUCCESS;
	}
	ptl_pt_index_t target = request->sequence % 2 ?
	    request->peer_second_pt_idx : request->peer_pt_idx;
	int ret = PtlPut(request->md_h, (uintptr_t)request->data_buffer,
	                 request->size, PRISM_ACK_TYPE,
	                 ctx.peer_procs[request->peer_rank], target, 0, 0, NULL, 0);
	if (ret != PTL_OK) {
		OPT_PERSISTENT_ERR("Buffered data put failed: %s",
		                   PtlToStr(ret, PTL_STR_ERROR));
		return PRISM_ERROR;
	}
	request->expected_md_ops++;
	*posted = 1;
	return PRISM_SUCCESS;
}

static int mpiext_persistent_progress_sends(void) {
	persistent_request_t **link = &ctx.pending_sends;
	while (*link) {
		persistent_request_t *request = *link;
		int posted;
		if (mpiext_persistent_buffered_try_send(request, &posted) != PRISM_SUCCESS)
			return PRISM_ERROR;
		if (posted) {
			*link = request->next_pending;
			request->next_pending = NULL;
			request->pending_send = 0;
		} else {
			link = &request->next_pending;
		}
	}
	return PRISM_SUCCESS;
}

static int mpiext_persistent_buffered_start(persistent_request_t *request) {
	if (request->active) {
		OPT_PERSISTENT_ERR("Cannot restart an active buffered request");
		return PRISM_ERROR;
	}
	if (request->init_state != READY &&
	    mpiext_persistent_finish_setup(request) != PRISM_SUCCESS)
		return PRISM_ERROR;
	request->active = 1;
	request->copied = 0;
	if (request->op_type == PERSISTENT_SEND) {
		int posted;
		if (mpiext_persistent_buffered_try_send(request, &posted) != PRISM_SUCCESS)
			return PRISM_ERROR;
		if (!posted) {
			request->pending_send = 1;
			request->next_pending = ctx.pending_sends;
			ctx.pending_sends = request;
		}
	}
	return PRISM_SUCCESS;
}

static int mpiext_persistent_counter_ready(ptl_handle_ct_t counter,
                                         uint64_t threshold, int *ready) {
	ptl_ct_event_t event;
	int ret = PtlCTGet(counter, &event);
	if (ret != PTL_OK || event.failure) {
		OPT_PERSISTENT_ERR("Buffered completion counter failed");
		return PRISM_ERROR;
	}
	*ready = event.success >= threshold;
	return PRISM_SUCCESS;
}

/* Called only from Wait/Waitall: arriving data must not alter the user's
 * receive buffer before this point. Return the credit strictly after memcpy. */
static int mpiext_persistent_buffered_test(persistent_request_t *request) {
	if (!request->active || request->pending_send) return PRISM_SUCCESS;
	int ready;
	if (request->op_type == PERSISTENT_RECV && !request->copied) {
		unsigned slot = request->sequence % 2;
		ptl_handle_ct_t counter = slot ? request->second_ct_h : request->le_ct_h;
		if (mpiext_persistent_counter_ready(counter, request->sequence / 2 + 1,
		                                   &ready) != PRISM_SUCCESS)
			return PRISM_ERROR;
		if (!ready) return PRISM_SUCCESS;
		if (request->size)
			memcpy(request->data_buffer,
			       (char *)request->bounce_buffer + slot * request->bounce_stride,
			       request->size);
		/* A zero-byte put increments the peer's LE counter once. It cannot
		 * race a previous credit's payload and requires no remote atomic. */
		int ret = PtlPut(request->md_h, 0, 0, PRISM_ACK_TYPE,
		                 ctx.peer_procs[request->peer_rank], request->peer_pt_idx,
		                 0, request->remote_flag_addr, NULL, 0);
		if (ret != PTL_OK) {
			OPT_PERSISTENT_ERR("Credit put failed: %s", PtlToStr(ret, PTL_STR_ERROR));
			return PRISM_ERROR;
		}
		request->expected_md_ops++;
		request->copied = 1;
	}
	if (mpiext_persistent_counter_ready(request->md_ct_h, request->expected_md_ops,
	                                   &ready) != PRISM_SUCCESS)
		return PRISM_ERROR;
	if (ready) {
		request->active = 0;
		request->sequence++;
	}
	return PRISM_SUCCESS;
}

static int mpiext_persistent_buffered_wait(persistent_request_t *request) {
	while (request->active) {
		if (mpiext_persistent_progress_sends() != PRISM_SUCCESS ||
		    mpiext_persistent_buffered_test(request) != PRISM_SUCCESS)
			return PRISM_ERROR;
	}
	return PRISM_SUCCESS;
}

static int mpiext_persistent_drain_credits(persistent_request_t *request) {
	if (request->op_type != PERSISTENT_SEND || !request->sequence)
		return PRISM_SUCCESS;
	/* Send Wait only promises source-buffer reuse. Before freeing the credit
	 * target, all matching receive Waits must have returned their credits. */
	int ready = 0;
	while (!ready) {
		if (mpiext_persistent_progress_sends() != PRISM_SUCCESS ||
		    mpiext_persistent_counter_ready(request->le_ct_h, request->sequence,
		                                   &ready) != PRISM_SUCCESS)
			return PRISM_ERROR;
	}
	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_data_transfer(persistent_request_t *request) {
	size_t local_offset = (size_t)request->data_buffer;
	size_t remote_offset = request->remote_data_addr;

	int ret = PtlPut(request->md_h, local_offset, request->size,
			 PRISM_ACK_TYPE, ctx.peer_procs[request->peer_rank],
			 request->peer_pt_idx, 0, remote_offset, request, 0);
	assert(ret == PTL_OK);
	request->expected_md_ops++;

	return PRISM_SUCCESS;
}

PERSISTENT_ALWAYS_INLINE static inline int
mpiext_persistent_start_flag_transfer(persistent_request_t *request) {
	ptl_ct_event_t event;
	/* Keep the source immutable until its local completion is consumed. */
	size_t local_offset = (size_t)&request->rtr_flag;
	size_t remote_offset = request->remote_flag_addr;
	int ret = PtlPut(request->md_h, local_offset, sizeof(int),
			 PRISM_ACK_TYPE, ctx.peer_procs[request->peer_rank],
			 request->peer_pt_idx, 0, remote_offset, request, 0);
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
	if (ctx.triggered) {
		uint64_t threshold = request->expected_le_ops + 1;
		int ret = PtlTriggeredPut(request->md_h,
		                         (uintptr_t)request->data_buffer, request->size,
		                         PRISM_ACK_TYPE, ctx.peer_procs[request->peer_rank],
		                         request->peer_pt_idx, 0, request->remote_data_addr,
		                         NULL, 0, request->le_ct_h, threshold);
		if (ret != PTL_OK) {
			OPT_PERSISTENT_ERR("PtlTriggeredPut failed: %s",
			                   PtlToStr(ret, PTL_STR_ERROR));
			return PRISM_ERROR;
		}
		request->expected_le_ops = threshold;
		request->expected_md_ops++;
		return PRISM_SUCCESS;
	}
	ptl_ct_event_t event;
	request->expected_le_ops++;
	int ret = PtlCTWait(request->le_ct_h, request->expected_le_ops, &event);
	assert(ret == PTL_OK);
	assert(event.failure == 0);

	return mpiext_persistent_start_data_transfer(request);
}

static inline int mpiext_persistent_finalize_send_core(
    persistent_request_t *request) {
	ptl_ct_event_t event;
	int ret = PtlCTWait(request->md_ct_h, request->expected_md_ops, &event);
	assert(ret == PTL_OK);
	assert(event.failure == 0);
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
	if (mpiext_persistent_finish_setup(request) != PRISM_SUCCESS)
		return PRISM_ERROR;

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
	if (mpiext_persistent_finish_setup(request) != PRISM_SUCCESS)
		return PRISM_ERROR;

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
	if (count < 0 || PMPI_Type_size(datatype, &size) != MPI_SUCCESS)
		return PRISM_ERROR;
	OPT_PERSISTENT_INFO("Called PSend_init");
	persistent_request_t *req = malloc(sizeof(persistent_request_t));

	assert(req != NULL);

	OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p", (void *)buffer);
	mpiext_persistent_reset_request(req);

	// check if av entry already exists
	req->id = combine32((uint32_t)tag, (uint32_t)ctx.my_world_rank);
	req->tag = tag;
	req->peer_rank = dst;
	req->data_buffer = (void *)buffer;
	req->op_type = PERSISTENT_SEND;

	req->size = (size_t)size * (size_t)count;

	mpiext_persistent_add_request_to_table(req->id, req);

	*request = (void *)req;
	OPT_PERSISTENT_INFO("[PSEND_INIT]: data buffer %p request: %p",
			    (void *)buffer, req);
	return mpiext_persistent_init_request(req);
}

int PRISM_Precv_init(void *buffer, int count, MPI_Datatype datatype, int src,
		     int tag, MPI_Comm communicator, PRISM_Request *request) {
	int size;
	if (count < 0 || PMPI_Type_size(datatype, &size) != MPI_SUCCESS)
		return PRISM_ERROR;
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

	req->size = (size_t)size * (size_t)count;

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
	if (ctx.buffered) return mpiext_persistent_buffered_start(request);
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
	if (ctx.buffered) return mpiext_persistent_buffered_start(request);
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
		if (!request_array[i]) continue;
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
	if (ctx.buffered) return mpiext_persistent_buffered_wait(request);
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
	if (ctx.buffered) {
		int remaining;
		do {
			remaining = 0;
			if (mpiext_persistent_progress_sends() != PRISM_SUCCESS)
				return PRISM_ERROR;
			/* Complete every ready receive, even if an earlier send in the
			 * array is waiting for credit. No copy happens outside this set. */
			for (int i = 0; i < count; ++i) {
				persistent_request_t *request = request_array[i];
				if (!request) continue;
				if (mpiext_persistent_buffered_test(request) != PRISM_SUCCESS)
					return PRISM_ERROR;
				remaining += request->active;
			}
		} while (remaining);
		return PRISM_SUCCESS;
	}
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
	if (ctx.buffered && (req->active ||
	    mpiext_persistent_drain_credits(req) != PRISM_SUCCESS))
		return PRISM_ERROR;
	OPT_PERSISTENT_INFO("Delte request with id %u", req->id);

	mpiext_persistent_remove_request_from_table(req);

	free(*request);
	*request = NULL;

	return PRISM_SUCCESS;
}
