#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct persistent_request *PRISM_Request;
#define PRISM_REQUEST_NULL NULL

int PRISM_Psend_init(const void *buffer, int count, MPI_Datatype datatype,
		     int dst, int tag, MPI_Comm communicator,
		     PRISM_Request *request);
int PRISM_Precv_init(void *buffer, int count, MPI_Datatype datatype, int src,
		     int tag, MPI_Comm communicator, PRISM_Request *request);
int PRISM_Start(PRISM_Request *request);
int PRISM_Startall(int count, PRISM_Request request_array[]);
int PRISM_Wait(PRISM_Request *request, MPI_Status *status);
int PRISM_Waitall(int count, PRISM_Request request_array[],
		  MPI_Status status_array[]);
int PRISM_Prequest_free(PRISM_Request *req);
int PRISM_Start_no_sync(PRISM_Request *request);
int PRISM_Startall_no_sync(int count, PRISM_Request request_array[]);
int PRISM_Init(void);
int PRISM_Finalize(void);

#ifdef __cplusplus
}
#endif
