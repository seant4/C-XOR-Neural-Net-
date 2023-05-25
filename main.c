#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "util.h"

#define features 2
#define hiddenNodes 2
#define outputNodes 1
#define samples 4

int forward_prop(double* hiddenLayerBias, double x_train[samples][features], 
                double hiddenWeights[features][hiddenNodes], double* hiddenLayer, double* outputLayer, double* outputLayerBias, 
                double outputWeights[hiddenNodes][outputNodes], double y_train[samples][1], int val){

    //Compute hidden layer
    double node = 0;
    for(int i = 0; i < hiddenNodes; i++){
        for(int k = 0; k < features; k++){
            node += (x_train[val][k] * hiddenWeights[k][i]) + hiddenLayerBias[i];
        }
        hiddenLayer[i] = sigmoid(node);
    }

    //Compute output layer
    node = 0;
    for(int i = 0; i < outputNodes; i++){
        for(int k = 0; k < hiddenNodes; k++){
            node += (hiddenLayer[k] * outputWeights[k][i]) + outputLayerBias[i];
        }
        outputLayer[i] = sigmoid(node);
    }
    return 0;
}

int backward_prop(double y_train[samples][1], double* outputLayer, double outputWeights[hiddenNodes][outputNodes], 
                double* hiddenLayer, double* delta, double* deltaHidden, int val){
    //Calculate error of output layer
    for(int j = 0; j < outputNodes; j++){
        double err = (y_train[val][j] - outputLayer[j]);
        delta[j] = err * dSigmoid(outputLayer[j]);
    }

    //Calculate error of hidden layer
    for(int j = 0; j < hiddenNodes; j++){
        double err = 0.0f;
        for(int k = 0; k < outputNodes; k++){
            err += delta[k] * outputWeights[j][k];
        }
        deltaHidden[j] = err * dSigmoid(hiddenLayer[j]);
    }
    return 0;
}

int update_params(double* outputLayerBias, double* delta, double alpha, double outputWeights[hiddenNodes][outputNodes], 
                double* hiddenLayer, double* hiddenLayerBias, double* deltaHidden,
                double hiddenWeights[features][hiddenNodes], double x_train[samples][features], int val){
    //Update the output layer
    for(int j = 0; j < outputNodes; j++){
        outputLayerBias[j] += delta[j] * alpha;
        for(int k = 0; k < hiddenNodes; k++){
            outputWeights[k][j] += hiddenLayer[k] * delta[j] * alpha;
        }
    }
    //Output the hidden layer
    for(int j = 0; j < hiddenNodes; j++){
        hiddenLayerBias[j] += deltaHidden[j] * alpha;
        for(int k = 0; k < features; k++){
            hiddenWeights[k][j] += x_train[val][k] * deltaHidden[j] * alpha;
        }
    }
}

//Initialize our random parameters
double init_params(){
    return ((double)rand()) / ((double)RAND_MAX);
}

int main(){
    const double alpha = 1.0f;

    //Initialize layers
    double hiddenLayer[hiddenNodes];
    double outputLayer[outputNodes];
    double hiddenLayerBias[hiddenNodes];
    double outputLayerBias[outputNodes];
    double hiddenWeights[features][hiddenNodes];
    double outputWeights[hiddenNodes][outputNodes];

    //Define our data
    double x_train[samples][features] = {
        {0.0f,0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
    };

    //Define our outputs
    double y_train[samples][outputNodes] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    //Fill in our layers
    for(int i = 0; i < features; i++){
        for(int j = 0; j < hiddenNodes; j++){
            hiddenWeights[i][j] = init_params();
        }
    }

    for(int i = 0; i < features; i++){
        for(int j = 0; j < outputNodes; j++){
            outputWeights[i][j] = init_params();
        }
    }

    for(int i = 0; i < outputNodes; i++){
        outputLayerBias[i] = init_params();
    }

    //Define random order of training to prevent over fitting
    int trainingSetOrder[] = {0,1,2,3};

    //Define epochs
    int epochs = 10000;

    //Begin training from epochs
    for(int i = 0; i < epochs; i++){
        //Train by each sample
        for(int x = 0; x < samples; x++){
            //Shuffle order of data
            shuffle(trainingSetOrder, samples);
            //Grab Sample to forward propogate
            int val = trainingSetOrder[x];
            //Forward Propogate
            forward_prop(hiddenLayerBias, x_train, hiddenWeights, 
                        hiddenLayer, outputLayer, outputLayerBias, outputWeights, y_train, val);
            //Define error arrays
            double delta[outputNodes];
            double deltaHidden[hiddenNodes];
            //Back propogate
            backward_prop(y_train, outputLayer, outputWeights, hiddenLayer, delta, deltaHidden, val);
            //Apply changes
            update_params(outputLayerBias, delta, alpha, outputWeights, hiddenLayer, hiddenLayerBias, deltaHidden, hiddenWeights, x_train, val);

        }
    }
    //Test model on all 4 XOR operations
    forward_prop(hiddenLayerBias, x_train, hiddenWeights, hiddenLayer, outputLayer, outputLayerBias, outputWeights, y_train, 0);
    print_matrix(outputNodes, 1, outputLayer);
    forward_prop(hiddenLayerBias, x_train, hiddenWeights, hiddenLayer, outputLayer, outputLayerBias, outputWeights, y_train, 1);
    print_matrix(outputNodes, 1, outputLayer);
    forward_prop(hiddenLayerBias, x_train, hiddenWeights, hiddenLayer, outputLayer, outputLayerBias, outputWeights, y_train, 2);
    print_matrix(outputNodes, 1, outputLayer);
    forward_prop(hiddenLayerBias, x_train, hiddenWeights, hiddenLayer, outputLayer, outputLayerBias, outputWeights, y_train, 3);
    print_matrix(outputNodes, 1, outputLayer);
}