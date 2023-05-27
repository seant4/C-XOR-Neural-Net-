#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "util.h"

#define FEATURES 2
#define SAMPLES 4
#define NEURONS 2
#define OUTPUTS 1

void init_params(int n1, int m1, int n2, int m2, double W1[n1][m1], double W2[n2][m1]){
    for(int i = 0; i < n1; i++){
        for(int j = 0; j < m1; j++){
            W1[i][j] = ((double)rand()) / ((double)RAND_MAX);
        }
    }

    for(int i = 0; i < n2; i++){
        for(int j = 0; j < m2; j++){
            W2[i][j] = ((double)rand()) / ((double)RAND_MAX);
        }
    }
}

void forward_prop(double W1[NEURONS][FEATURES], double W2[OUTPUTS][NEURONS], double X[FEATURES][SAMPLES], double Z1[NEURONS][SAMPLES], 
                double A1[NEURONS][SAMPLES], double Z2[OUTPUTS][SAMPLES], double A2[OUTPUTS][SAMPLES]){

    matrix_dot(2,2,2,4,W1,X,Z1);
    for(int i = 0; i < NEURONS; i++){
        for(int j = 0; j < SAMPLES; j++){
            A1[i][j] = sigmoid(Z1[i][j]);
        }
    }

    matrix_dot(1,2,2,4,W2,A1,Z2);
    for(int i = 0; i < OUTPUTS; i++){
        for(int j = 0; j < SAMPLES; j++){
            A2[i][j] = sigmoid(Z2[i][j]);
        }
    }
}

void back_prop(double W1[NEURONS][FEATURES], double W2[OUTPUTS][NEURONS], double Z1[NEURONS][SAMPLES], double A1[NEURONS][SAMPLES], 
                double Z2[OUTPUTS][SAMPLES], double A2[OUTPUTS][SAMPLES], double dW1[NEURONS][FEATURES], double dW2[OUTPUTS][NEURONS], 
                double dZ1[NEURONS][SAMPLES], double dZ2[OUTPUTS][SAMPLES], double X[FEATURES][SAMPLES], double Y[OUTPUTS][SAMPLES]){

    for(int i = 0; i < OUTPUTS; i++){
        for(int j = 0; j < SAMPLES; j++){
            dZ2[i][j] = A2[i][j] - Y[i][j];
        }
    }
    double A1T[SAMPLES][NEURONS];
    transposeMatrix(NEURONS, SAMPLES, A1, A1T);
    matrix_dot(OUTPUTS, SAMPLES, SAMPLES, NEURONS, dZ2, A1T, dW2);
    for(int i = 0; i < OUTPUTS; i++){
        for(int j = 0; j < NEURONS; j++){
            dW2[i][j] = dW2[i][j] / SAMPLES;
        }
    }
    double W2T[NEURONS][OUTPUTS];
    transposeMatrix(NEURONS, OUTPUTS, W2, W2T);
    matrix_dot(NEURONS, OUTPUTS, OUTPUTS, SAMPLES, W2T, dZ2, dZ1);
    for(int i = 0; i < NEURONS; i++){
        for(int j = 0; j < SAMPLES; j++){
            dZ1[i][j] = dZ1[i][j] * (A1[i][j]*(1-A1[i][j]));
        }
    }
    double XT[SAMPLES][FEATURES];
    transposeMatrix(FEATURES, SAMPLES, X, XT);
    matrix_dot(NEURONS, SAMPLES, SAMPLES, FEATURES, dZ1, XT, dW1);
    for(int i = 0; i < NEURONS; i++){
        for(int j = 0; j < FEATURES; j++){
            dW1[i][j] = dW1[i][j] / SAMPLES;
        }
    }
}

void update_params(double W1[NEURONS][FEATURES], double W2[OUTPUTS][NEURONS], double dW1[NEURONS][FEATURES], double dW2[OUTPUTS][NEURONS], double alpha){
    for(int i = 0; i < OUTPUTS; i++){
        for(int j = 0; j < NEURONS; j++){
            W2[i][j] = W2[i][j] - alpha * dW2[i][j];
        }
    }
    for(int i = 0; i < NEURONS; i++){
        for(int j = 0; j < FEATURES; j++){
            W1[i][j] = W1[i][j] - alpha * dW1[i][j]; 
        }
    }
}

void forward_prop_test(double W1[NEURONS][FEATURES], double W2[OUTPUTS][NEURONS], double X[FEATURES][1], double Z1[NEURONS][1], 
                double A1[NEURONS][1], double Z2[OUTPUTS][1], double A2[OUTPUTS][1]){

    matrix_dot(2,2,2,1,W1,X,Z1);
    for(int i = 0; i < NEURONS; i++){
        for(int j = 0; j < 1; j++){
            A1[i][j] = sigmoid(Z1[i][j]);
        }
    }

    matrix_dot(1,2,2,1,W2,A1,Z2);
    for(int i = 0; i < OUTPUTS; i++){
        for(int j = 0; j < 1; j++){
            A2[i][j] = sigmoid(Z2[i][j]);
        }
    }
}

int main(){
    //Define weight matricies
    double W1[NEURONS][FEATURES];
    double W2[OUTPUTS][NEURONS];
    double Z1[NEURONS][SAMPLES];
    double A1[NEURONS][SAMPLES];
    double Z2[OUTPUTS][SAMPLES];
    double A2[OUTPUTS][SAMPLES];

    //Define training data
    double x_train[FEATURES][SAMPLES] = {{0,0,1,1},{0,1,0,1}};
    double y_train[OUTPUTS][SAMPLES] = {{0,1,1,0}};

    //Define delta matricies
    double dW1[NEURONS][FEATURES];
    double dW2[OUTPUTS][NEURONS];
    double dZ1[NEURONS][SAMPLES];
    double dA1[NEURONS][SAMPLES];
    double dZ2[OUTPUTS][SAMPLES];
    double dA2[OUTPUTS][SAMPLES];

    init_params(2,2,1,2,W1,W2);

    for(int i = 0; i < 10000; i++){
        forward_prop(W1,W2,x_train,Z1,A1,Z2,A2);
        back_prop(W1,W2,Z1,A1,Z2,A2,dW1,dW2,dZ1,dZ2,x_train,y_train);
        update_params(W1,W2,dW1,dW2,0.5);
    }
    print_matrix(1,4,A2);
    return 0;
}