#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank = -1, comm_sz = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	if (comm_sz != 2) {
		if (rank == 0) fprintf(stderr, "Este programa requiere exactamente 2 procesos.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	long N = 100000;
	if (argc > 1) {
		long tmp = atol(argv[1]);
		if (tmp > 0) N = tmp;
	}

	int msg = 0;
	const int tag = 0;

	clock_t c0 = clock();
	for (long i = 0; i < N; ++i) {
		if (rank == 0) {
			MPI_Send(&msg, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
			MPI_Recv(&msg, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&msg, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&msg, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		}
	}
	clock_t c1 = clock();
	double cpu_secs = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;

	MPI_Barrier(MPI_COMM_WORLD);
	double t0 = MPI_Wtime();
	for (long i = 0; i < N; ++i) {
		if (rank == 0) {
			MPI_Send(&msg, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
			MPI_Recv(&msg, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&msg, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&msg, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		}
	}
	double t1 = MPI_Wtime();
	double wall_secs = t1 - t0;

	if (rank == 0) {
		printf("Iteraciones: %ld\n", N);
		printf("clock()  (CPU time)   = %.9f s\n", cpu_secs);
		printf("MPI_Wtime (wall time) = %.9f s\n", wall_secs);
		printf("Latencia media ida+vuelta (wall) = %.3f us\n", (wall_secs / (double)N) * 1e6);
	}

	MPI_Finalize();
	return 0;
}
