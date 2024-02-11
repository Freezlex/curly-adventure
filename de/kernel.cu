
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>

const int POP_SIZE = 500;
const int NUM_OF_DIMENSIONS = 100;
const int MAX_ITER = NUM_OF_DIMENSIONS * pow(10, 4);
__constant__ float Pc = 0.7;
__constant__ float Pm = 0.01;
const float phi = 3.1415;
const int SELECTED_OBJ_FUNC = 4;

__device__ curandState_t devStates[POP_SIZE];

struct Individual {
    float position[NUM_OF_DIMENSIONS];
    float fitness;
};

__device__ float getRandom() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    return curand_uniform(&devStates[tid]);
}

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
// parametre 1 individu avec ses positions
__device__ float host_fitness_function(float* x) {
    float res = 0;
    float somme = 0;
    float produit = 1; // Initialize to 1 to avoid multiplication with 0

    switch (SELECTED_OBJ_FUNC) {
    case 0: {
        float y1 = 1 + (x[0] - 1) / 4;
        float yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;

        res += pow(sin(3.1415 * y1), 2);

        for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
            float y = 1 + (x[i] - 1) / 4;
            float yp = 1 + (x[i + 1] - 1) / 4;
            res += pow(y - 1, 2) * (1 + 10 * pow(sin(3.1415 * yp), 2)) + pow(yn - 1, 2);
        }
        break;
    }
    case 1: {
        for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
            float zi = x[i] - 0;
            res += pow(zi, 2) - 10 * cos(2 * 3.1415 * zi) + 10;
        }
        res -= 330;
        break;
    }
    case 2:
        for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
            float zi = x[i] - 0 + 1;
            float zip1 = x[i + 1] - 0 + 1;
            res += 100 * (pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
        }
        res += 390;
        break;
    case 3:
        for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
            float zi = x[i] - 0;
            somme += pow(zi, 2) / 4000;
            produit *= cos(zi / pow(i + 1, 0.5));
        }
        res = somme - produit + 1 - 180;
        break;
    case 4:
        for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
            float zi = x[i] - 0;
            res += pow(zi, 2);
        }
        res -= 450;
        break;
    }

    return res;
}


__global__ void init_population_kernel(Individual* population, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POP_SIZE) {
        Individual individual;
        for (int j = 0; j < NUM_OF_DIMENSIONS; ++j) {
            individual.position[j] = getRandom();
        }
        individual.fitness = host_fitness_function(individual.position);
        population[idx] = individual;
    }
}


__device__ int select_individual(curandState_t* state) {
    return curand(state) % POP_SIZE;
}

__global__ void selection_kernel(curandState_t* states, int* index1, int* index2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        *index1 = select_individual(&states[idx]);
        do {
            *index2 = select_individual(&states[idx]);
        } while (*index1 == *index2);
    }
}
__device__ void crossover(Individual& offspring1, Individual& offspring2, const Individual& parent1, const Individual& parent2) {
    float p = getRandom();
    if (p < Pc) {
        for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
            if (getRandom() < 0.5) {
                offspring1.position[i] = parent1.position[i];
                offspring2.position[i] = parent2.position[i];
            }
            else {
                offspring1.position[i] = parent2.position[i];
                offspring2.position[i] = parent1.position[i];
            }
        }
    }
    else {
        offspring1 = parent1;
        offspring2 = parent2;
    }
    offspring1.fitness = host_fitness_function(offspring1.position);
    offspring2.fitness = host_fitness_function(offspring2.position);
}

__device__ void mutation(Individual& individual) {
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        if (getRandom() < Pm) {
            individual.position[i] = getRandom();
        }
    }
    individual.fitness = host_fitness_function(individual.position);
}

__global__ void crossover_mutation_kernel(Individual* population, int* index1, int* index2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POP_SIZE) {
        Individual offspring1, offspring2;
        crossover(offspring1, offspring2, population[*index1], population[*index2]);
        mutation(offspring1);
        mutation(offspring2);
        population[*index1] = offspring1;
        population[*index2] = offspring2;
    }
}

__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < POP_SIZE) {
        curand_init(seed, id, 0, &state[id]);
    }
}

void mainFunction() {
    srand(static_cast<unsigned>(time(nullptr)));

    // Allocation de mémoire sur le GPU pour la population
    Individual* dev_population;
    cudaMalloc(&dev_population, POP_SIZE * sizeof(Individual));

    // Initialisation de l'état du générateur de nombres aléatoires
    curandState* devStates;
    cudaMalloc(&devStates, POP_SIZE * sizeof(curandState));

    // Initialisation des états curand sur le GPU
    setup_kernel << <(POP_SIZE + 1023) / 1024, 1024 >> > (devStates, time(NULL));
    cudaDeviceSynchronize(); // Assurez-vous que l'initialisation est terminée

    // Initialiser la population sur le GPU
    init_population_kernel << <(POP_SIZE + 1023) / 1024, 1024 >> > (dev_population, devStates);
    cudaDeviceSynchronize(); // Assurez-vous que l'initialisation de la population est terminée

    int gen = 0;
    int evaluations = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    int* dev_index1;
    int* dev_index2;
    cudaMalloc(&dev_index1, sizeof(int));
    cudaMalloc(&dev_index2, sizeof(int));

    while (evaluations < MAX_ITER) {
        // Sélection, crossover, et mutation sur le GPU
        selection_kernel << <1, 1 >> > (devStates, dev_index1, dev_index2);
        crossover_mutation_kernel << <(POP_SIZE + 1023) / 1024, 1024 >> > (dev_population, dev_index1, dev_index2);

        evaluations += POP_SIZE; // Mettre à jour le nombre d'évaluations
        ++gen;
    }

    // Copier la population du GPU vers le CPU pour l'analyse finale
    std::vector<Individual> population(POP_SIZE);
    cudaMemcpy(population.data(), dev_population, POP_SIZE * sizeof(Individual), cudaMemcpyDeviceToHost);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // Trouver le meilleur individu
    Individual best_individual = *std::min_element(population.begin(), population.end(),
        [](const Individual& ind1, const Individual& ind2) {
            return ind1.fitness < ind2.fitness;
        });

    // Trier la position du meilleur individu pour l'affichage
    std::sort(std::begin(best_individual.position), std::end(best_individual.position));

    // Affichage des résultats
    std::cout << "Meilleure fitness : " << std::setprecision(std::numeric_limits<float>::max_digits10) << best_individual.fitness << std::endl;
    std::cout << "Meilleure position (triee) : ";
    for (float val : best_individual.position) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    int minutes = duration.count() / 60;
    int seconds = duration.count() % 60;
    std::cout << "Temps ecoule : " << minutes << " minutes " << seconds << " secondes" << std::endl;
    std::cout << "Total des evaluations : " << evaluations << std::endl;
    std::cout << "Generations : " << gen << std::endl;

    // Libérer la mémoire du GPU
    cudaFree(dev_population);
    cudaFree(devStates);
    cudaFree(dev_index1);
    cudaFree(dev_index2);
}



int main() {
    mainFunction();
    return 0;
}

// ===============================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <math_functions.h>
#include <cfloat>

#include "../includes/kernel.h"


__device__ curandState_t devStates[POP_SIZE];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
__global__ void initRandomStates(unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &devStates[tid]);
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
 * Initializes the population and writes it into global memory.
 * This kernel requires random numbers.
*/
__global__ void kernelPopInit(float *devPos, float *devObjectiveValues) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < POP_SIZE * NUM_OF_DIMENSIONS) {
        devPos[i] = getRandom();
        float x[NUM_OF_DIMENSIONS];
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            x[j] = devPos[i + j];
        }
        devObjectiveValues[i / NUM_OF_DIMENSIONS] = fitness_function(x);
    }
}

/**
 * Runs on the GPU, called from the CPU.
 * Evaluates the objective function values of population
 * members as per the problem being solved and writes them into
 * global memory. The evaluation of each population member can
 * take up one or more threads. This kernel needs to read the
 * population to be evaluated from global memory
 * 
 * Also count the number of evaluations, and stop the algorithm when the maxEvals is reached
*/
__global__ void kernelEval(float *devTrial, float *devObjectiveValues) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < POP_SIZE) {
        float x[NUM_OF_DIMENSIONS];
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            x[j] = devTrial[(i * NUM_OF_DIMENSIONS) + j];
        }

        devObjectiveValues[i] = fitness_function(x);
    }
}

/**
 * Runs on the GPU, called from the CPU.
 * Prepares the mutually exclusive indices of randomly
 * sampled population members for each target vector to generate
 * the mutant vector. The indices are structured into a matrix and
 * written into global memory. This kernel requires random numbers.
*/
__global__ void kernelMutationPrep(float *devIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < POP_SIZE * 4) {
        devIndices[i] = getRandom() * POP_SIZE;
    }
}

/**
 * Runs on the GPU, called from the CPU.
 * Performs the DE mutation to generate mutant vectors,
 * which are then written into global memory. This kernel needs to
 * read the current population from global memory and requires
 * random numbers.
*/
__global__ void kernelMutation(float *devPos, float *devMutant, float *devIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < POP_SIZE * NUM_OF_DIMENSIONS) {
        int idx = i % POP_SIZE;

        float baseVector[NUM_OF_DIMENSIONS];
        float differenceVector[NUM_OF_DIMENSIONS];
        float mutantVector[NUM_OF_DIMENSIONS];

        int r1 = (int)devIndices[idx];
        int r2 = (int)devIndices[idx + POP_SIZE];
        int r3 = (int)devIndices[idx + 2 * POP_SIZE];

        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            baseVector[j] = devPos[r1 * NUM_OF_DIMENSIONS + j];
            differenceVector[j] = devPos[r2 * NUM_OF_DIMENSIONS + j] - devPos[r3 * NUM_OF_DIMENSIONS + j];
            mutantVector[j] = baseVector[j] + F * differenceVector[j];
        }

        // Write the mutant vector back to global memory
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            devMutant[i + j] = mutantVector[j];
        }
    }
}

/**
 * Runs on the GPU, called from the CPU.
 * Performs the DE crossover to generate trial vectors,
 * which are written into global memory. This kernel needs to read
 * the current population and mutant vectors generated in kernel(M)
 * from global memory and requires random numbers.
*/
__global__ void kernelCrossover(float *devPos, float *devMutant, float *devTrial, float *devIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < POP_SIZE * NUM_OF_DIMENSIONS) {
        int idx = i % POP_SIZE;

        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            if (getRandom() <= CR || j == (int)devIndices[idx + 3 * POP_SIZE]) {
                // Crossover is applied
                devTrial[i + j] = devMutant[i + j];
            } else {
                // No crossover, copy from the target vector
                devTrial[i + j] = devPos[i + j];
            }
        }
    }
}

/**
 * Runs on the GPU, called from the CPU.
 * Performs the objective function values comparison
 * between trial vectors and their corresponding target vectors to
 * form the population of the next generation, which is then written
 * into global memory. This kernel needs to read the objective
 * function values of trial and target vectors, trial vectors and the
 * current population from global memory.
*/
__global__ void kernelReplacement(float *devPos, float *devTrial, float *devObjectiveValues) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < POP_SIZE * NUM_OF_DIMENSIONS) {
        int idx = i % POP_SIZE;
        int targetIdx = idx * NUM_OF_DIMENSIONS;

        // Compare objective values
        if (devObjectiveValues[idx] < devObjectiveValues[targetIdx]) {
            // Replace the target vector with the trial vector
            for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
                devPos[targetIdx + j] = devTrial[i + j];
            }
        }
    }
}

/**
 * Runs on the GPU, called from the CPU.
 * Find bestest solution in the current population.
*/
__global__ void kernelFindBest(float *objectiveValues, float *positions, float *bestObjective, float *bestPosition) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < POP_SIZE) {
        if (objectiveValues[i] < *bestObjective) {
            *bestObjective = objectiveValues[i];
            for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
                bestPosition[j] = positions[i * NUM_OF_DIMENSIONS + j];
            }
        }
    }
}

extern "C" void cuda_de(float *positions, float *gBest)
{
    int size = POP_SIZE * NUM_OF_DIMENSIONS;

    // Declare all the arrays on the device
    float *devPos;

    // Arrays for mutation and crossover
    float *devMutant;
    float *devTrial;
    float *devIndices;

    // Array for objective values
    float *devObjectiveValues;

    float *devBestObjective;
    float *devBestPosition;

    // Memory allocation
    gpuErrchk(cudaMalloc((void**)&devPos, sizeof(float) * size));
    gpuErrchk(cudaMalloc((void**)&devMutant, sizeof(float) * size));
    gpuErrchk(cudaMalloc((void**)&devTrial, sizeof(float) * size));
    gpuErrchk(cudaMalloc((void**)&devIndices, sizeof(float) * POP_SIZE * 4));
    gpuErrchk(cudaMalloc((void**)&devObjectiveValues, sizeof(float) * POP_SIZE));
    gpuErrchk(cudaMalloc((void**)&devBestObjective, sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&devBestPosition, sizeof(float) * NUM_OF_DIMENSIONS));

    // Thread & Block number
    int threadsNum = 32;
    int blocksNum = ceil(size / static_cast<float>(threadsNum));


    #pragma region Logs_for_device_debugging
    printf("Threads: %d\n", threadsNum);
    printf("Blocks: %d\n", blocksNum);

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Needed memory: %zu bytes\n", size * sizeof(float));
    printf("Free memory: %zu bytes\n", free);
    printf("Total memory: %zu bytes\n", total);

    size_t requiredMemory = size * sizeof(float);
    if (requiredMemory > free) {
        printf("Not enough memory on the GPU for the array\n");
        return;
    }
    #pragma endregion


    float hostBestObjective = FLT_MAX;
    float hostBestPosition[NUM_OF_DIMENSIONS] = {0};

    gpuErrchk(cudaMemcpy(devBestObjective, &hostBestObjective, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devBestPosition, hostBestPosition, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice));
    
    // Copy particle datas from host to device
    gpuErrchk(cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice));

    // Initialize random states
    initRandomStates<<<blocksNum, threadsNum>>>(time(NULL));

    // Initialise la population
    kernelPopInit<<<blocksNum, threadsNum>>>(devPos, devObjectiveValues);

    // Init best objective and position for the init population
    kernelFindBest<<<blocksNum, threadsNum>>>(devObjectiveValues, devTrial, devBestObjective, devBestPosition);


    // DE main loop
    for (int iter = 0; iter < (MAX_ITER - POP_SIZE) / POP_SIZE; iter++) {
        // Mutation preparation
        kernelMutationPrep<<<blocksNum, threadsNum>>>(devIndices);

        // Mutation
        kernelMutation<<<blocksNum, threadsNum>>>(devPos, devMutant, devIndices);

        // Crossover
        kernelCrossover<<<blocksNum, threadsNum>>>(devPos, devMutant, devTrial, devIndices);

        // Evaluation of trial vectors
        kernelEval<<<blocksNum, threadsNum>>>(devTrial, devObjectiveValues);

        // Replacement
        kernelReplacement<<<blocksNum, threadsNum>>>(devPos, devTrial, devObjectiveValues);

        // Check if the current result is better than the best result
        kernelFindBest<<<blocksNum, threadsNum>>>(devObjectiveValues, devTrial, devBestObjective, devBestPosition);
    }

    printf("Start waiting for GPU to finish...\n");

    // Synchronize threads before checking result
    cudaDeviceSynchronize();

    printf("FINISHED !\n");

    // Copy the final results back to host
    gpuErrchk(cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(gBest, devBestPosition, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost));

    // Cleanup
    gpuErrchk(cudaFree(devPos));
    gpuErrchk(cudaFree(devMutant));
    gpuErrchk(cudaFree(devTrial));
    gpuErrchk(cudaFree(devIndices));
    gpuErrchk(cudaFree(devObjectiveValues));
    gpuErrchk(cudaFree(devBestObjective));
    gpuErrchk(cudaFree(devBestPosition));
}

