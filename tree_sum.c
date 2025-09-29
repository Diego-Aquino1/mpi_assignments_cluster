#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int is_power_of_two(int value) {
	return value > 0 && (value & (value - 1)) == 0;
}

static long long reduce_power_of_two(long long local_value, int my_rank, int comm_sz, MPI_Comm comm) {
	int step = 1;
	while (step < comm_sz) {
		int partner;
		if ((my_rank % (2 * step)) == 0) {
			partner = my_rank + step;
			if (partner < comm_sz) {
				long long recv_val = 0;
				MPI_Recv(&recv_val, 1, MPI_LONG_LONG, partner, 0, comm, MPI_STATUS_IGNORE);
				local_value += recv_val;
			}
		} else if ((my_rank % (2 * step)) == step) {
			partner = my_rank - step;
			MPI_Send(&local_value, 1, MPI_LONG_LONG, partner, 0, comm);
			break;
		}
		step <<= 1;
	}
	return local_value;
}

static long long reduce_general(long long local_value, int my_rank, int comm_sz, MPI_Comm comm) {
	/* Plegado previo hacia la mayor potencia de dos <= comm_sz */
	int p2 = 1;
	while ((p2 << 1) <= comm_sz) p2 <<= 1;

	if (my_rank >= p2) {
		int dst = my_rank - p2;
		MPI_Send(&local_value, 1, MPI_LONG_LONG, dst, 0, comm);
		return 0;
	} else {
		int src = my_rank + p2;
		if (src < comm_sz) {
			long long recv_val = 0;
			MPI_Recv(&recv_val, 1, MPI_LONG_LONG, src, 0, comm, MPI_STATUS_IGNORE);
			local_value += recv_val;
		}
	}

	return reduce_power_of_two(local_value, my_rank, p2, comm);
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int my_rank = -1, comm_sz = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	long long local_value = (long long) my_rank;
	if (argc > 1) {
		local_value = atoll(argv[1]) + (long long) my_rank;
	}

	long long global_sum = 0;
	if (is_power_of_two(comm_sz)) {
		global_sum = reduce_power_of_two(local_value, my_rank, comm_sz, MPI_COMM_WORLD);
	} else {
		global_sum = reduce_general(local_value, my_rank, comm_sz, MPI_COMM_WORLD);
	}

	if (my_rank == 0) {
		printf("Suma global = %lld (comm_sz=%d)\n", global_sum, comm_sz);
	}

	MPI_Finalize();
	return 0;
}
