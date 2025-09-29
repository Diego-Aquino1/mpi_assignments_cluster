#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int is_power_of_two(int value) {
	return value > 0 && (value & (value - 1)) == 0;
}

static long long reduce_butterfly_pow2(long long local_value, int my_rank, int comm_sz, MPI_Comm comm) {
	for (int s = 0; (1 << s) < comm_sz; ++s) {
		int partner = my_rank ^ (1 << s);
		if ((my_rank & (1 << s)) == 0) {
			long long recv_val = 0;
			MPI_Recv(&recv_val, 1, MPI_LONG_LONG, partner, s, comm, MPI_STATUS_IGNORE);
			local_value += recv_val;
			MPI_Send(&local_value, 1, MPI_LONG_LONG, partner, s, comm);
		} else {
			MPI_Send(&local_value, 1, MPI_LONG_LONG, partner, s, comm);
			long long recv_val = 0;
			MPI_Recv(&recv_val, 1, MPI_LONG_LONG, partner, s, comm, MPI_STATUS_IGNORE);
			local_value += recv_val;
		}
	}
	return local_value; /* todos los procesos terminan con la suma global */
}

static long long reduce_butterfly_general(long long local_value, int my_rank, int comm_sz, MPI_Comm comm) {
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

	return reduce_butterfly_pow2(local_value, my_rank, p2, comm);
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
		global_sum = reduce_butterfly_pow2(local_value, my_rank, comm_sz, MPI_COMM_WORLD);
	} else {
		global_sum = reduce_butterfly_general(local_value, my_rank, comm_sz, MPI_COMM_WORLD);
	}

	if (my_rank == 0) {
		printf("Suma global (butterfly) = %lld (comm_sz=%d)\n", global_sum, comm_sz);
	}

	MPI_Finalize();
	return 0;
}
