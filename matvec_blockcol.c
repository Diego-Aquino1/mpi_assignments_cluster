#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void generate_matrix_vector(double* A, double* x, int n) {
	for (int i = 0; i < n; ++i) {
		x[i] = 1.0;
		for (int j = 0; j < n; ++j) {
			A[(size_t)i * (size_t)n + (size_t)j] = (double)(i + j + 1);
		}
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank = -1, comm_sz = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	int n = 8;
	if (argc > 1) {
		int tmp = atoi(argv[1]);
		if (tmp > 0) n = tmp;
	}

	if (n % comm_sz != 0) {
		if (rank == 0) {
			fprintf(stderr, "Error: n (%d) debe ser divisible por comm_sz (%d)\n", n, comm_sz);
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	const int cols_per_proc = n / comm_sz;
	const int local_mat_elems = n * cols_per_proc;

	double* A_local = (double*) malloc((size_t)local_mat_elems * sizeof(double));
	double* x_local = (double*) malloc((size_t)cols_per_proc * sizeof(double));
	double* y_partial = (double*) malloc((size_t)n * sizeof(double));
	if (!A_local || !x_local || !y_partial) {
		fprintf(stderr, "Rank %d: fallo de memoria\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	double* A = NULL;
	double* x = NULL;
	if (rank == 0) {
		A = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
		x = (double*) malloc((size_t)n * sizeof(double));
		if (!A || !x) {
			fprintf(stderr, "Rank 0: fallo de memoria para A/x\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		generate_matrix_vector(A, x, n);
	}

	MPI_Scatter(x, cols_per_proc, MPI_DOUBLE,
	            x_local, cols_per_proc, MPI_DOUBLE,
	            0, MPI_COMM_WORLD);

	double* sendbuf = NULL;
	if (rank == 0) {
		sendbuf = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
		if (!sendbuf) {
			fprintf(stderr, "Rank 0: fallo de memoria para sendbuf\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		for (int p = 0; p < comm_sz; ++p) {
			int start_col = p * cols_per_proc;
			double* dst = sendbuf + (size_t)p * (size_t)local_mat_elems;
			for (int i = 0; i < n; ++i) {
				const double* src_row = A + (size_t)i * (size_t)n + (size_t)start_col;
				double* dst_row = dst + (size_t)i * (size_t)cols_per_proc;
				for (int j = 0; j < cols_per_proc; ++j) dst_row[j] = src_row[j];
			}
		}
	}

	MPI_Scatter(sendbuf, local_mat_elems, MPI_DOUBLE,
	            A_local, local_mat_elems, MPI_DOUBLE,
	            0, MPI_COMM_WORLD);

	if (rank == 0) {
		free(sendbuf);
		free(A);
		free(x);
	}

	for (int i = 0; i < n; ++i) y_partial[i] = 0.0;
	for (int i = 0; i < n; ++i) {
		const double* arow = A_local + (size_t)i * (size_t)cols_per_proc;
		double acc = 0.0;
		for (int j = 0; j < cols_per_proc; ++j) {
			acc += arow[j] * x_local[j];
		}
		y_partial[i] = acc;
	}

	const int rows_per_proc = n / comm_sz;
	double* y_local_rows = (double*) malloc((size_t)rows_per_proc * sizeof(double));
	if (!y_local_rows) {
		fprintf(stderr, "Rank %d: fallo de memoria y_local_rows\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Reduce_scatter_block(y_partial, y_local_rows, rows_per_proc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	double* y_final = NULL;
	if (rank == 0) {
		y_final = (double*) malloc((size_t)n * sizeof(double));
		if (!y_final) {
			fprintf(stderr, "Rank 0: fallo de memoria y_final\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	MPI_Gather(y_local_rows, rows_per_proc, MPI_DOUBLE,
	           y_final,      rows_per_proc, MPI_DOUBLE,
	           0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Resultado y = A*x (n=%d):\n", n);
		for (int i = 0; i < n; ++i) {
			printf("%.3f\n", y_final[i]);
		}
		free(y_final);
	}

	free(y_local_rows);
	free(y_partial);
	free(x_local);
	free(A_local);

	MPI_Finalize();
	return 0;
}
