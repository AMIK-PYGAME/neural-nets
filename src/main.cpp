#include <iostream>
#include "neural_net.h"
#include<Eigen/Dense>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>


using namespace std;

int main() {
    
    vector<int> structure = {784,16,16,10};
    ANN* net = new ANN(structure);
    
    ifstream mnist_dataset("data\\mnist_train.csv");//open this file

    auto start = chrono::high_resolution_clock::now();
    net->train(mnist_dataset,60000,10);
    auto stop = chrono::high_resolution_clock::now();
    auto time = chrono::duration_cast<chrono::milliseconds>(stop-start);
    cout<<time.count();
    std::cin.get();//waits for you to press enter before closing external console
    return 0;
}