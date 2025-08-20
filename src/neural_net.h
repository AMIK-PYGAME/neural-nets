#pragma once
#include<Eigen/Dense>
#include"layer.h"
#include"math.h"
#include <fstream>

class ANN{
public:
    vector<layer*> layers={};
    vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> all_weights={};//size will be layers.size()-1

    ANN(std::vector<int> setuplayers){
        //the setuplayers vector contains the structure such as {2,5,1}
        for(int i=0;i<setuplayers.size();i++){
            layers.push_back(new layer(setuplayers[i]));
        }
        
        //setup the weights
        for(int i=0;i<setuplayers.size()-1;i++){
            all_weights.push_back(setuprandomweights(layers[i],layers[i+1]));
        }
   
    };
    
    //setup
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> setuprandomweights(layer* firstlayer,layer* secondlayer){
            
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> weights;
        
        //the no of columns is the no of weights a neuron connects to from the prev layer
        //the no of rows is the no of neurons in the current layer
        weights.resize(secondlayer->no_of_neurons,firstlayer->no_of_neurons);
        std::random_device rd;
        std::mt19937 gen(rd());//this is a mersenne twister ... a pseudo -random number generator
        std::uniform_real_distribution<float> dist(-1.0,1.0);
        float scale = sqrt(2.0f / firstlayer->no_of_neurons);
        for(int rows=0;rows<weights.rows();rows++){
            for(int cols = 0;cols<weights.cols();cols++){
                weights(rows,cols) = dist(gen);
            };
        };
        weights *= scale;
        return weights;

    };
    
    //loading data 
    void Readmnistdata(ifstream &dataset,Eigen::Matrix<float,Eigen::Dynamic,1> &ExpectedOutput,Eigen::Matrix<float,Eigen::Dynamic,1> &Inputs){
        
        string line;
        getline(dataset,line);//read the next line
        stringstream ss(line);//convert the string into a stream
        string Activationstring;
        float valuefloat;
        
        for(int i=0;i<Inputs.rows()+1;i++){
            getline(ss,Activationstring,',');//read indivudual value
            valuefloat = stof(Activationstring);//convert the value to a float type
            //create the expected outputs vector from the data given by the first column
            if(i==0){
                for(int j=0;j<ExpectedOutput.rows();j++){
                    ExpectedOutput(j, 0) = (j == static_cast<int>(valuefloat)) ? 1.0f : 0.0f;
                }
            }//else just create an input vector for the rest of the coulumns
            
            else{
                valuefloat /= 255;//normalize the Activations within a range of 0-1
                Inputs(i-1,0)=valuefloat;
            };
        };
    };

    //the no of rows in the input vector has to be equal to the number of perceptrons
    Eigen::Matrix<float,Eigen::Dynamic,1>  Feedforward(Eigen::Matrix<float,Eigen::Dynamic,1> Inputs){
            layers[0]->Activations = Inputs;

            for(int i=0;i<all_weights.size();i++){
                //just multiply the previous layers Activations with weight , add biases and reluuuuuuuuu
                layers[i+1]->Z = (all_weights[i]*(layers[i]->Activations))+(layers[i+1]->Biases);
                

                if(i!=all_weights.size()-1)layers[i+1]->Activations = ReLu(layers[i+1]->Z);
                else layers[i+1]->Activations = Sigmoid(layers[i+1]->Z);
            };
            //technically you only need the activations of the last layer sunce everything else just uses relu
            return layers.back()->Activations;

    };

    float evaluate(Eigen::Matrix<float,Eigen::Dynamic,1> ExpectedOutputs,Eigen::Matrix<float,Eigen::Dynamic,1> guessed_Outputs){
        int ex_label,guessed_label;
        ExpectedOutputs.maxCoeff(&ex_label);
        guessed_Outputs.maxCoeff(&guessed_label);

        if(guessed_label==ex_label) return 1;
        else return 0;

    };

    void train(ifstream &dataset,int no_of_examples,int epochs){
        Eigen::Matrix<float,Eigen::Dynamic,1> expected_outputs(layers.back()->no_of_neurons,1);
        Eigen::Matrix<float,Eigen::Dynamic,1> Inputs(layers[0]->no_of_neurons,1);
        float learning_rate = 0.001;
        int ExOneEpoch = no_of_examples/epochs; //no of examples in one epoch ... used for sgd as well as evaluation
        float no_of_correct_answers=0;
        for(int epoch=0;epoch<epochs;epoch++){
            for(int example = 0;example<ExOneEpoch;example++)
            {
                Readmnistdata(dataset,expected_outputs,Inputs);
                auto outputs = Feedforward(Inputs);
                no_of_correct_answers+=evaluate(expected_outputs,outputs);
                Backprop(expected_outputs,learning_rate);
            }
            cout<<epoch<<" : "<<100*(no_of_correct_answers/ExOneEpoch)<<" %"<<"\n";
            no_of_correct_answers =0;
        }
    };
    
    //the thing that makes it learn
    void Backprop(Eigen::Matrix<float,Eigen::Dynamic,1> ExpectedOutputs,float learning_rate = 0.01f){
        //the last layer
        Eigen::Matrix<float,Eigen::Dynamic,1> loss = 2*(layers.back()->Activations-ExpectedOutputs); 
        Eigen::Matrix<float,Eigen::Dynamic,1> delta = SigmoidPrime(layers.back()->Z).cwiseProduct(loss);

        //multiply delta withe the activations from second to last layer to get the weights gradient
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> grad_w = delta*layers[layers.size()-2]->Activations.transpose();
        Eigen::Matrix<float,Eigen::Dynamic,1> grad_b = delta;
        
        all_weights.back()-= learning_rate*grad_w;
        layers.back()->Biases-=learning_rate*grad_b;

        //need the for loop to be 2 less than the current no of weights,since it backpropagets through the hidden layers
        for(int backindex=all_weights.size()-2;backindex>=0;backindex--){
            //delta = weights from the next layer * sigprime of activations from the next layer 
            delta = (all_weights[backindex+1].transpose()*delta).cwiseProduct((ReluPrime(layers[backindex+1]->Z)));
            grad_b = delta;
            grad_w = delta*layers[backindex]->Activations.transpose();
            all_weights[backindex]-=learning_rate*grad_w;
            layers[backindex+1]->Biases-=learning_rate*grad_b;
        
        }

    };  

};
