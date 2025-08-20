#pragma once
#include<Eigen/Dense>
#include<algorithm>
#define _USE_MATH_DEFINES // Required for some compilers
#include <cmath>


//for hidden layers
Eigen::Matrix<float,Eigen::Dynamic,1> ReLu(Eigen::Matrix<float,Eigen::Dynamic,1> x){
    for(int i=0;i<x.rows();i++){
        x(i,0)=(float)std::max((double)0.0f,(double)x(i,0));
    }
    
    return x;
}

Eigen::Matrix<float,Eigen::Dynamic,1> ReluPrime(Eigen::Matrix<float,Eigen::Dynamic,1> x){
    for(int i=0;i<x.rows();i++){
        x(i,0)=(x(i,0)==0)?0:1;
    }
    
    return x;
};

//for last layer
Eigen::Matrix<float,Eigen::Dynamic,1> Sigmoid(Eigen::Matrix<float,Eigen::Dynamic,1> values){
    for(int i=0;i<values.rows();i++){
        values(i,0) = 1/(1+pow(M_E,-values(i,0)));
    }
    return values;
};

//for gradient calculations
Eigen::Matrix<float,Eigen::Dynamic,1> SigmoidPrime(Eigen::Matrix<float,Eigen::Dynamic,1> values){
    values = Sigmoid(values);
    return values-(values.cwiseProduct(values));
};

//not in use currently too hard too implement for a beginner will take a look later
Eigen::Matrix<float, Eigen::Dynamic,1> SoftMaxDerivative(Eigen:: Matrix<float,Eigen::Dynamic,1> values){
    float sum = 0;
    float value = 0;
    for(int i=0;i<values.rows();i++){
        sum += pow(M_E,values(i,0));
    };
    for(int i=0;i<values.rows();i++){
        value = pow(M_E,values(i,0))/sum;//this is the actual softmax function
        value *= (1-value);//this accounts for the derivative
        values(i,0)=value;
    }

    return values;
    
};
