#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

	int q = (int)(sqrt((double)comm_sz) + 0.5);
	if (q * q != comm_sz) {
		if (rank == 0) fprintf(stderr, "Error: comm_sz debe ser un cuadrado perfecto\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (n % q != 0) {
		if (rank == 0) fprintf(stderr, "Error: sqrt(comm_sz) debe dividir n\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	const int b = n / q;
	const int my_row = rank / q;
	const int my_col = rank % q;
	const int is_diag = (my_row == my_col);

	MPI_Comm row_comm, col_comm, diag_comm = MPI_COMM_NULL;
	MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);
	if (is_diag) {
		MPI_Comm_split(MPI_COMM_WORLD, 0, my_row, &diag_comm);
	} else {
		MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &diag_comm);
	}

	double* A_local = (double*) malloc((size_t)b * (size_t)b * sizeof(double));
	double* x_block = (double*) malloc((size_t)b * sizeof(double));
	double* y_part  = (double*) malloc((size_t)b * sizeof(double));
	if (!A_local || !x_block || !y_part) {
		fprintf(stderr, "Rank %d: fallo de memoria\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	double* A = NULL; double* x = NULL;
	if (rank == 0) {
		A = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
		x = (double*) malloc((size_t)n * sizeof(double));
		if (!A || !x) { fprintf(stderr, "Rank 0: sin memoria para A/x\n"); MPI_Abort(MPI_COMM_WORLD, 1);}    
		generate_matrix_vector(A, x, n);
	}

	if (rank == 0) {
		for (int r = 0; r < q; ++r) {
			for (int c = 0; c < q; ++c) {
				int dst = r * q + c;
				double* block = (double*) malloc((size_t)b * (size_t)b * sizeof(double));
				for (int i = 0; i < b; ++i) {
					const double* src_row = A + (size_t)(r * b + i) * (size_t)n + (size_t)(c * b);
					double* dst_row = block + (size_t)i * (size_t)b;
					for (int j = 0; j < b; ++j) dst_row[j] = src_row[j];
				}
				if (dst == 0) {
					for (int i = 0; i < b * b; ++i) A_local[i] = block[i];
					free(block);
				} else {
					MPI_Send(block, b * b, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD);
					free(block);
				}
			}
		}
	} else {
		MPI_Recv(A_local, b * b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	if (rank == 0) {
		for (int k = 0; k < q; ++k) {
			int dst = k * q + k;
			if (dst == 0) {
				for (int i = 0; i < b; ++i) x_block[i] = x[k * b + i];
			} else {
				MPI_Send(x + (size_t)k * (size_t)b, b, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
			}
		}
	} else if (is_diag) {
		MPI_Recv(x_block, b, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	int col_rank = -1; MPI_Comm_rank(col_comm, &col_rank);
	MPI_Bcast(x_block, b, MPI_DOUBLE, /*root=*/my_col, col_comm);

	for (int i = 0; i < b; ++i) y_part[i] = 0.0;
	for (int i = 0; i < b; ++i) {
		const double* arow = A_local + (size_t)i * (size_t)b;
		double acc = 0.0;
		for (int j = 0; j < b; ++j) acc += arow[j] * x_block[j];
		y_part[i] = acc;
	}

	double* y_row = NULL;
	if (is_diag) y_row = (double*) malloc((size_t)b * sizeof(double));
	MPI_Reduce(y_part, is_diag ? y_row : NULL, b, MPI_DOUBLE, MPI_SUM, /*root=*/my_row, row_comm);

	double* y_final = NULL;
	int diag_rank = -1, diag_size = 0;
	if (diag_comm != MPI_COMM_NULL) {
		MPI_Comm_rank(diag_comm, &diag_rank);
		MPI_Comm_size(diag_comm, &diag_size);
		if (diag_rank == 0) y_final = (double*) malloc((size_t)n * sizeof(double));
		MPI_Gather(y_row, b, MPI_DOUBLE,
		           y_final, b, MPI_DOUBLE,
		           0, diag_comm);
	}

	if (rank == 0) {
		printf("Resultado y = A*x (n=%d):\n", n);
		for (int i = 0; i < n; ++i) printf("%.3f\n", y_final[i]);
		free(y_final);
	}

	free(y_row);
	free(y_part);
	free(x_block);
	free(A_local);
	if (rank == 0) { free(A); free(x); }

	if (row_comm != MPI_COMM_NULL) MPI_Comm_free(&row_comm);
	if (col_comm != MPI_COMM_NULL) MPI_Comm_free(&col_comm);
	if (diag_comm != MPI_COMM_NULL) MPI_Comm_free(&diag_comm);

	MPI_Finalize();
	return 0;
}
