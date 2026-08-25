#include <mpi.h>
#include <prism.h>

int main(int argc, char *argv[]){

	MPI_Init(&argc, &argv);
	PRISM_Init();
	int x = 1;
	PRISM_Request req;

	PRISM_Psend_init(&x,1,MPI_INT,1,1,MPI_COMM_WORLD,&req);

	PRISM_Prequest_free(&req);

	PRISM_Finalize();
	MPI_Finalize();
	return 0;
}
