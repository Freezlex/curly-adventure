#include "cuda_de.h"
#include <iomanip>

int main(int argc, char** argv) {
    Particle* gBest;

    std::cout << "Type \t Time \t  \t Minimum\n";

    clock_t begin = clock();
    cuda_de(gBest);
    clock_t end = clock();

    std::cout << "GPU \t " << (double)(end - begin) / CLOCKS_PER_SEC << "\t";
    std::cout << std::fixed << std::setprecision(9) << host_fitness_function(&gBest->fitness) << std::endl;

    delete[] gBest;

    return 0;
}
