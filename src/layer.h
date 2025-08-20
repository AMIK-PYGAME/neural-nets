#pragma once
#include <Eigen/Dense>
#include <random>
using namespace std;

class layer{
public:
    int no_of_neurons;
    Eigen::Matrix<float,Eigen::Dynamic,1> Biases;//dynamic means the size of the matrix can be changed
    Eigen::Matrix<float,Eigen::Dynamic,1> Activations;
    Eigen::Matrix<float,Eigen::Dynamic,1> Z;
    

    layer(int No_of_neurons){
        no_of_neurons = No_of_neurons;
        
        Biases = Eigen::Matrix<float, Eigen::Dynamic, 1>(no_of_neurons, 1);//create the biases vector
        Activations = Eigen::Matrix<float, Eigen::Dynamic, 1>(no_of_neurons, 1);//create the values vector

        // Random number generation setup
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for(int n=0; n<no_of_neurons; ++n)
            Biases(n,0) = dist(gen); // generate biases
    };
};