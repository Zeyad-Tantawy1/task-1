#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Eigen;

int countRows(const string& path) {
    ifstream file(path);
    string l;
    int rowCount = 0;
    while (getline(file, l)) {
        rowCount++;
    }
    return rowCount;
}

MatrixXd loadCSV(const string& path, int rows, int cols) {
    ifstream file(path);
    MatrixXd data(rows, cols);
    string l;
    int row = 0;

    while (getline(file, l)) {
        stringstream l_s(l);
        string cell;
        int col = 0;
        while (getline(l_s, cell, ',')) {
            data(row, col) = stod(cell);
            col++;
        }
        row++;
    }

    return data;
}

// Function to normalize features
MatrixXd normalize(const MatrixXd& m) {
    MatrixXd X_norm = m;
    for (int i = 0; i < m.cols(); i++) {
        X_norm.col(i) = (m.col(i).array() - m.col(i).mean()) / m.col(i).std();
    }
    return X_norm;
}

// Function to compute the cost
double computeCost(const MatrixXd& X, const MatrixXd& y, const MatrixXd& theta) {
    MatrixXd predictions = X * theta;
    MatrixXd errors = predictions - y;
    return (errors.array().square().sum()) / (2 * X.rows());
}

// Function to perform gradient descent
MatrixXd gradientDescent(const MatrixXd& X, const MatrixXd& y, MatrixXd theta, double alpha, int iterations) {
    int m = X.rows();
    for (int i = 0; i < iterations; i++) {
        MatrixXd predictions = X * theta;
        MatrixXd errors = predictions - y;
        theta = theta - (alpha / m) * (X.transpose() * errors);
    }
    return theta;
}

int main() {
    string filePath = "house_prices.csv";

    // Count rows in the CSV file
    int rows = countRows(filePath);
    int cols = 4; // Number of columns in the dataset (price, sqft, bedrooms, bathrooms)
    
    // Load dataset
    MatrixXd data = loadCSV(filePath, rows, cols);

    // Separate features and target variable
    MatrixXd X = data.leftCols(cols - 1);
    MatrixXd y = data.rightCols(1);

    // Normalize features
    X = normalize(X);

    // Add a column of ones to X (intercept term)
    MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << MatrixXd::Ones(X.rows(), 1), X;

    // Initialize theta
    MatrixXd theta = MatrixXd::Zero(X_b.cols(), 1);

    // Hyperparameters
    double alpha = 0.01;
    int iterations = 1000;

    // Perform gradient descent
    theta = gradientDescent(X_b, y, theta, alpha, iterations);

    // Output the resulting theta
    cout << "Theta: " << endl << theta << endl;

    // Predict using the trained model
    MatrixXd predictions = X_b * theta;
    cout << "Predictions: " << endl << predictions << endl;

    // Compute the final cost
    double cost = computeCost(X_b, y, theta);
    cout << "Final Cost: " << cost << endl;

    return 0;
}
