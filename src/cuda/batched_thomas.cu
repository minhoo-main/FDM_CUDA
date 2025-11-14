#include <cuda_runtime.h>
#include <stdio.h>

namespace ELSPricer {
namespace CUDA {

/**
 * Batched Thomas Algorithm CUDA Kernel
 *
 * Solves batch_size independent tridiagonal systems in parallel
 * Each thread block handles one system
 *
 * Input:
 *   lower: Lower diagonal [N-1] (shared across all systems)
 *   diag: Main diagonal [N] (shared across all systems)
 *   upper: Upper diagonal [N-1] (shared across all systems)
 *   rhs: Right-hand sides [batch_size x N] (different for each system)
 *
 * Output:
 *   solution: Solutions [batch_size x N]
 */
__global__ void batchedThomasKernel(
    const double* __restrict__ lower,
    const double* __restrict__ diag,
    const double* __restrict__ upper,
    const double* __restrict__ rhs,
    double* __restrict__ solution,
    int N,
    int batch_size)
{
    int bid = blockIdx.x;  // batch index
    if (bid >= batch_size) return;

    // Each block processes one tridiagonal system
    extern __shared__ double shared_mem[];

    double* c_prime = shared_mem;           // [N-1]
    double* d_prime = &shared_mem[N - 1];   // [N]

    const double* rhs_b = &rhs[bid * N];
    double* sol_b = &solution[bid * N];

    // Forward sweep (single thread per block for simplicity)
    // For large N, can parallelize this too
    if (threadIdx.x == 0) {
        // First row
        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs_b[0] / diag[0];

        // Middle rows
        for (int i = 1; i < N - 1; ++i) {
            double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
            c_prime[i] = upper[i] / denom;
            d_prime[i] = (rhs_b[i] - lower[i - 1] * d_prime[i - 1]) / denom;
        }

        // Last row
        int i = N - 1;
        double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        d_prime[i] = (rhs_b[i] - lower[i - 1] * d_prime[i - 1]) / denom;

        // Backward substitution
        sol_b[N - 1] = d_prime[N - 1];
        for (int i = N - 2; i >= 0; --i) {
            sol_b[i] = d_prime[i] - c_prime[i] * sol_b[i + 1];
        }
    }
}

/**
 * Optimized Batched Thomas with parallelized forward/backward sweep
 * Uses parallel reduction techniques
 */
__global__ void batchedThomasKernelOptimized(
    const double* __restrict__ lower,
    const double* __restrict__ diag,
    const double* __restrict__ upper,
    const double* __restrict__ rhs,
    double* __restrict__ solution,
    int N,
    int batch_size)
{
    int bid = blockIdx.x * blockDim.y + threadIdx.y;
    if (bid >= batch_size) return;

    extern __shared__ double shared_mem[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Shared memory layout per batch item
    double* c_prime = &shared_mem[threadIdx.y * (2 * N - 1)];
    double* d_prime = &shared_mem[threadIdx.y * (2 * N - 1) + N - 1];

    const double* rhs_b = &rhs[bid * N];
    double* sol_b = &solution[bid * N];

    // Forward sweep (parallelized)
    if (tid == 0) {
        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs_b[0] / diag[0];
    }
    __syncthreads();

    for (int i = 1 + tid; i < N - 1; i += stride) {
        // Wait for previous iteration
        while (i > 0 && c_prime[i - 1] == 0.0 && d_prime[i - 1] == 0.0) {
            // Spin wait (not ideal but simple)
        }

        double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        c_prime[i] = upper[i] / denom;
        d_prime[i] = (rhs_b[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    if (tid == 0) {
        int i = N - 1;
        double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        d_prime[i] = (rhs_b[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }
    __syncthreads();

    // Backward substitution
    if (tid == 0) {
        sol_b[N - 1] = d_prime[N - 1];
        for (int i = N - 2; i >= 0; --i) {
            sol_b[i] = d_prime[i] - c_prime[i] * sol_b[i + 1];
        }
    }
}

/**
 * Host wrapper for batched Thomas algorithm
 */
void batchedThomas(
    const double* d_lower,
    const double* d_diag,
    const double* d_upper,
    const double* d_rhs,
    double* d_solution,
    int N,
    int batch_size)
{
    // Kernel configuration
    int threadsPerBlock = 128;
    int blocksPerGrid = batch_size;

    // Shared memory size: (N-1) for c_prime + N for d_prime
    size_t sharedMemSize = (2 * N - 1) * sizeof(double);

    batchedThomasKernel<<<blocksPerGrid, 1, sharedMemSize>>>(
        d_lower, d_diag, d_upper, d_rhs, d_solution, N, batch_size
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * Apply boundary conditions kernel
 */
__global__ void applyBoundaryConditionsKernel(
    double* V,
    int N1,
    int N2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // S1 = 0: V = 0
    if (idx < N2) {
        V[0 * N2 + idx] = 0.0;
    }

    // S2 = 0: V = 0
    if (idx < N1) {
        V[idx * N2 + 0] = 0.0;
    }

    // S1 = S1_max: Linear extrapolation
    if (idx < N2) {
        V[(N1 - 1) * N2 + idx] = 2.0 * V[(N1 - 2) * N2 + idx] - V[(N1 - 3) * N2 + idx];
    }

    // S2 = S2_max: Linear extrapolation
    if (idx < N1) {
        V[idx * N2 + (N2 - 1)] = 2.0 * V[idx * N2 + (N2 - 2)] - V[idx * N2 + (N2 - 3)];
    }
}

void applyBoundaryConditions(double* d_V, int N1, int N2) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (std::max(N1, N2) + threadsPerBlock - 1) / threadsPerBlock;

    applyBoundaryConditionsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_V, N1, N2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Boundary conditions kernel error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * Early Redemption GPU Kernel
 *
 * Applies early redemption condition for Step-Down ELS
 * If min(S1/S1_0, S2/S2_0) >= barrier, V = principal + coupon (forced redemption)
 */
__global__ void applyEarlyRedemptionKernel(
    double* __restrict__ V,
    const double* __restrict__ S1,
    const double* __restrict__ S2,
    double S1_0,
    double S2_0,
    double barrier,
    double principal,
    double coupon,
    int N1,
    int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N1 && j < N2) {
        double s1_pct = S1[i] / S1_0;
        double s2_pct = S2[j] / S2_0;
        double worst = (s1_pct < s2_pct) ? s1_pct : s2_pct;

        if (worst >= barrier) {
            // Early redemption is mandatory, not optional
            double redemption_value = principal + coupon;
            int idx = i * N2 + j;
            V[idx] = redemption_value;
        }
    }
}

/**
 * Transpose kernel (for S1 direction batched solve)
 */
__global__ void transposeKernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int rows,
    int cols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}

void transpose(const double* d_input, double* d_output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Transpose kernel error: %s\n", cudaGetErrorString(err));
    }
}

void applyEarlyRedemption(
    double* d_V,
    const double* d_S1,
    const double* d_S2,
    double S1_0,
    double S2_0,
    double barrier,
    double principal,
    double coupon,
    int N1,
    int N2)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N2 + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    applyEarlyRedemptionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_V, d_S1, d_S2, S1_0, S2_0, barrier, principal, coupon, N1, N2
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Early redemption kernel error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace CUDA
} // namespace ELSPricer
