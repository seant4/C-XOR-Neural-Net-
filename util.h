#include <stdio.h>
#include <math.h>

#define FEATURES 2
#define SAMPLES 4

// Function to transpose a matrix
void transposeMatrix(int m, int n, double matrix[m][n], double transposed[n][m]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

void sumColumns(int rows, int columns, double matrix[rows][columns], double sums[]) {
    for (int j = 0; j < columns; j++) {
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            sum += matrix[i][j];
        }
        sums[j] = sum;
    }
}

int matrix_dot(int m1, int n1, int m2, int n2, double a[m1][n1], double b[m2][n2], double result[m1][n2]){
    for(int i = 0; i < m1; i++){
        for(int j = 0; j < n2; j++){
            result[i][j] = 0;
            for(int k = 0; k < m2; k++){
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return 0;
}

int matrix_vector_add(int m, int n, double a[m][n], double v[n]){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            a[m][n] += v[m];
        }
    }
    return 0;
}

int matrix_vector_subtract(int m, int n, double a[m][n], double v[n], double result[m][n]){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            result[m][n] = a[m][n] - v[m];
        }
    }
    return 0;
}
int print_matrix(int m, int n, double a[m][n]){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("Matrix at [%d][%d]: %f\n", i, j , a[i][j]);
        }
    }
    return 0;
}

double relu(double x){
    if(x > 0){
        return x;
    }
    return 0;
}

double relu_deriv(double x){
    if(x > 0){
        return 1;
    }else{
        return 0;
    }
}

double sigmoid(double x){
    return (1 / (1 + exp(-1 * x)));
}

double sigmoid_deriv(double x){
    return (sigmoid(x) * (1 - sigmoid(x)));
}

void findLargestIndexForEachColumn(int rows, int cols, double matrix[rows][cols], double result[cols]) {
    for (int j = 0; j < cols; j++) {
        int maxVal = matrix[0][j];
        int maxIndex = 0;

        for (int i = 1; i < cols; i++) {
            if (matrix[i][j] > maxVal) {
                maxVal = matrix[i][j];
                maxIndex = i;
            }
        }

        result[j] = maxIndex;
    }
}

void softmax(double *input, int input_len) {
  double m = -INFINITY;
  for (int i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  double sum = 0.0;
  for (int i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  double offset = m + logf(sum);
  for (int i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}