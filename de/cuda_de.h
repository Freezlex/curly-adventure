#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>


// Constantes
/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
const int SELECTED_OBJ_FUNC = 0;
const int NUM_OF_PARTICLES = 100;
const int NUM_OF_DIMENSIONS = 50;
const int MAX_ITER = NUM_OF_DIMENSIONS * pow(10, 4);
const float cF = 0;
const float mF = 0;
const float phi = 3.1415;

struct Particle {
    float position[NUM_OF_DIMENSIONS];
    float fitness;
};

// Les 3 fonctions très utiles
float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);

// Fonction externe qui va tourner sur le GPU
extern "C" void cuda_de(Particle *gBest);

