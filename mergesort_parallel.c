#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int cmp_int(const void* a, const void* b) {
	int x = *(const int*)a;
	int y = *(const int*)b;
	return (x > y) - (x < y);
}

static int* merge_sorted(const int* a, int na, const int* b, int nb) {
	int* out = (int*) malloc((size_t)(na + nb) * sizeof(int));
	if (!out) return NULL;
	int i = 0, j = 0, k = 0;
	while (i < na && j < nb) {
		if (a[i] <= b[j]) out[k++] = a[i++];
		else out[k++] = b[j++];
	}
	while (i < na) out[k++] = a[i++];
	while (j < nb) out[k++] = b[j++];
	return out;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank = -1, comm_sz = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	int n = 32;
	if (rank == 0) {
		if (argc > 1) {
			int tmp = atoi(argv[1]);
			if (tmp > 0) n = tmp;
		}
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (n % comm_sz != 0) {
		if (rank == 0) fprintf(stderr, "Error: n (%d) debe ser divisible por comm_sz (%d)\n", n, comm_sz);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int local_n = n / comm_sz;
	int* local = (int*) malloc((size_t)local_n * sizeof(int));
	if (!local) { fprintf(stderr, "Rank %d: sin memoria\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

	unsigned int seed = (unsigned int) (time(NULL) ^ (rank * 2654435761u));
	for (int i = 0; i < local_n; ++i) local[i] = (int)(rand_r(&seed) % (n * 4 + 1));
	qsort(local, (size_t)local_n, sizeof(int), cmp_int);

	int* gathered = NULL;
	if (rank == 0) gathered = (int*) malloc((size_t)n * sizeof(int));
	MPI_Gather(local, local_n, MPI_INT, gathered, local_n, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("Listas locales (concatenadas por rank):\n");
		for (int p = 0; p < comm_sz; ++p) {
			printf("Rank %d: ", p);
			for (int i = 0; i < local_n; ++i) printf("%d ", gathered[p * local_n + i]);
			printf("\n");
		}
	}

	int curr_n = local_n;
	int* curr = local;
	for (int step = 1; step < comm_sz; step <<= 1) {
		if ((rank % (2 * step)) == 0) {
			int partner = rank + step;
			if (partner < comm_sz) {
				int recv_n = 0;
				MPI_Recv(&recv_n, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				int* recv_buf = (int*) malloc((size_t)recv_n * sizeof(int));
				if (!recv_buf) { fprintf(stderr, "Rank %d: sin memoria\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
				MPI_Recv(recv_buf, recv_n, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				int* merged = merge_sorted(curr, curr_n, recv_buf, recv_n);
				free(curr);
				free(recv_buf);
				curr = merged;
				curr_n += recv_n;
			}
		} else if ((rank % (2 * step)) == step) {
			int partner = rank - step;
			MPI_Send(&curr_n, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
			MPI_Send(curr, curr_n, MPI_INT, partner, 0, MPI_COMM_WORLD);
			free(curr);
			curr = NULL;
			break;
		}
	}

	if (rank == 0) {
		printf("\nLista global ordenada (n=%d):\n", n);
		for (int i = 0; i < curr_n; ++i) printf("%d ", curr[i]);
		printf("\n");
	}

	free(curr);
	free(gathered);
	MPI_Finalize();
	return 0;
}
