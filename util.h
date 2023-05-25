#include <stdio.h>
#include <math.h>

int print_matrix(int m, int n, double a[m][n]){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("Matrix at [%d][%d]: %f\n", i, j , a[i][j]);
        }
    }
    return 0;
}

double sigmoid(double x){
    return 1 / (1+exp(-1 * x));
}

double dSigmoid(double x){
    return x * (1-x);
}

void shuffle(int *array, int n){
    if(n > 1){
        for(int i = 0; i < n - 1; i++){
            int j = 1 + rand() / (RAND_MAX / (n-i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}