#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <math_functions.h>

#include "cuda_de.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ curandState_t devStates[NUM_OF_PARTICLES];

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
/**
 * Runs on the GPU, called from the GPU.
*/
__device__ float fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;

    switch (SELECTED_OBJ_FUNC)  {
        case 0: 
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS-1] - 1)/4;

            res += pow(sin(phi*y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        case 1: 
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
        
        case 2:
            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 180; 
            break;
        case 4:
            for(int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }

    return res;
}

/**
 * Runs on the GPU, called from the CPU.
 * Initializes the random states for each thread.
 */
__global__ void KInitRandomStates(unsigned int seed) {

    // Get thread id
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Init random seed
    curand_init(seed, i, 0, &devStates[i]);
}

/**
 * Runs on the GPU, called from the GPU.
 * Generates a random float using the curand library.
 */
__device__ float getRandom() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    return curand_uniform(&devStates[tid]);
}

/**
 * Runs on the GPU, called from the CPU.
 * Find bestest solution in the current population.
*/
__global__ void KFindBest(Particle* pop, Particle &gBest) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // avoid an out of bound for the array 
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
        return;

    if(pop[i].fitness < gBest.fitness)
        gBest = pop[i];
}

/**
 * Init population through kernel to avoid memory transfert between host and device
 */
__global__ void KInitPopulation(Particle* pop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // avoid an out of bound for the array 
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
        return;
    
    Particle particle;
    for(int j=0;i<NUM_OF_DIMENSIONS;j++)
        particle.position[j] = getRandom();

    particle.fitness = host_fitness_function(particle.position);
    pop[i] = particle;
}

__global__ void KMutateParticle(Particle* pop, Particle* mutants) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure tid is within bounds
    if(tid >= NUM_OF_PARTICLES)
        return;

    // Select random indices for mutation
    int r1 = tid;
    int r2 = (tid + 1) % NUM_OF_PARTICLES;
    int r3 = (tid + 2) % NUM_OF_PARTICLES;

    // Generate a mutant solution
    for(int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        mutants[tid].position[i] = pop[r1].position[i] + mF * (pop[r2].position[i] - pop[r3].position[i]);
    }
}

__global__ void KCrossoverParticle(Particle* pop, Particle* mutants, Particle* offspring) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure tid is within bounds
    if(tid >= NUM_OF_PARTICLES)
        return;

    // Perform crossover operation
    for(int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        if(getRandom() < cF || i == rand() % NUM_OF_DIMENSIONS) {
            offspring[tid].position[i] = mutants[tid].position[i];
        } else {
            offspring[tid].position[i] = pop[tid].position[i];
        }
    }
}

__global__ void KEvalAndReplacePop(Particle* pop, Particle* offspring) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure tid is within bounds
    if(tid >= NUM_OF_PARTICLES)
        return;

    // Evaluate fitness of offspring
    offspring[tid].fitness = host_fitness_function(offspring[tid].position);

    // Replace population member if offspring is better
    if(offspring[tid].fitness < pop[tid].fitness) {
        pop[tid] = offspring[tid];
    }
}

/**
 * Runs on the GPU, called from the CPU.
 * Find bestest solution in the current population.
*/
__global__ void kernelFindBest(float *objectiveValues, float *positions, float *bestObjective, float *bestPosition) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUM_OF_PARTICLES) {
        if (objectiveValues[i] < *bestObjective) {
            *bestObjective = objectiveValues[i];
            for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
                bestPosition[j] = positions[i * NUM_OF_DIMENSIONS + j];
            }
        }
    }
}


extern "C" void cuda_de(Particle* gBest)
{
    // TODO : Clean above
    int size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;
    int threadAmnt = 32;
    int blocksAmnt = ceil(size / threadAmnt);

    // Particles that build ou population
    Particle* devPop;
    Particle* devGBest;
    Particle* devOffspring;
    // Random states for each thread
    Particle* devRandPop;

    // GPU Memory allocation
    gpuErrchk(cudaMalloc(&devPop, NUM_OF_PARTICLES * sizeof(Particle)));
    gpuErrchk(cudaMalloc(&devRandPop, NUM_OF_PARTICLES * sizeof(Particle)));
    gpuErrchk(cudaMalloc(&devGBest, sizeof(Particle)));
    gpuErrchk(cudaMalloc(&devOffspring, NUM_OF_PARTICLES * sizeof(Particle)));

    KInitRandomStates<<<blocksAmnt, threadAmnt>>>(time(NULL));
    cudaDeviceSynchronize();

    KInitPopulation<<<blocksAmnt, threadAmnt>>>(devPop);
    KInitPopulation<<<blocksAmnt, threadAmnt>>>(devRandPop);
    cudaDeviceSynchronize();

    KFindBest<<<blocksAmnt, threadAmnt>>>(devPop, *devGBest);

    int iter;
    for (iter = 0; iter < MAX_ITER; iter++)
    {
        KMutateParticle<<<blocksAmnt, threadAmnt>>>(devPop, devRandPop);

        // Crossover
        KCrossoverParticle<<<blocksAmnt, threadAmnt>>>(devPop, devRandPop, devOffspring);

        // Replacement
        KEvalAndReplacePop<<<blocksAmnt, threadAmnt>>>(devPop, devOffspring);

        // Check if the current result is better than the best result
        KFindBest<<<blocksAmnt, threadAmnt>>>(devPop, *devGBest);
    }

    printf("Start waiting for GPU to finish...\n");
    
    cudaDeviceSynchronize();

    // Copy values back to host
    gpuErrchk(cudaMemcpy(devGBest, gBest, sizeof(Particle), cudaMemcpyDeviceToHost));


    printf("Best fitness : %zu", gBest->fitness);
    std::cout << gBest->fitness << std::endl;
    
    // cleanup
    cudaFree(devGBest);
    cudaFree(devPop);
    cudaFree(devRandPop);
    cudaFree(devOffspring);
}