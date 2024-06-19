#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>

using namespace std;
using namespace Eigen;

// Function to load data from a CSV file
MatrixXd loadCSV(const string& path, int rows, int cols) {
    ifstream file(path);
    MatrixXd data(rows, cols);
    string line;
    int row = 0;

    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        int col = 0;
        while (getline(lineStream, cell, ',')) {
            data(row, col) = stod(cell);
            col++;
        }
        row++;
    }

    return data;
}

// Function to calculate Euclidean distance between two vectors
double calculateDistance(const VectorXd& v1, const VectorXd& v2) {
    return sqrt((v1 - v2).array().square().sum());
}

// K-means clustering function
pair<MatrixXd, VectorXi> kmeans(const MatrixXd& data, int k, int maxIterations = 100) {
    int n = data.rows();
    int d = data.cols();
    
    // Randomly initialize centroids
    MatrixXd centroids = data.topRows(k);
    
    VectorXi labels = VectorXi::Zero(n);
    MatrixXd newCentroids = MatrixXd::Zero(k, d);

    for (int iter = 0; iter < maxIterations; iter++) {
        // Assign clusters
        for (int i = 0; i < n; i++) {
            double minDist = numeric_limits<double>::max();
            int bestCluster = 0;

            for (int j = 0; j < k; j++) {
                double dist = calculateDistance(data.row(i), centroids.row(j));
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }
            labels(i) = bestCluster;
        }

        // Recalculate centroids
        newCentroids.setZero();
        VectorXi counts = VectorXi::Zero(k);

        for (int i = 0; i < n; i++) {
            int cluster = labels(i);
            newCentroids.row(cluster) += data.row(i);
            counts(cluster)++;
        }

        for (int j = 0; j < k; j++) {
            if (counts(j) != 0) {
                newCentroids.row(j) /= counts(j);
            }
        }

        // Check for convergence
        if ((centroids - newCentroids).norm() < 1e-4) {
            break;
        }

        centroids = newCentroids;
    }

    return {centroids, labels};
}

int main() {
    string filePath = "purchase_history.csv";

    // Count rows in the CSV file
    int rows = 100;  // Adjust based on your dataset
    int cols = 5;    // Adjust based on your dataset

    // Load dataset
    MatrixXd data = loadCSV(filePath, rows, cols);

    // Number of clusters
    int k = 3;

    // Run K-means clustering
    auto [centroids, labels] = kmeans(data, k);

    // Output the resulting centroids and labels
    cout << "Centroids: " << endl << centroids << endl;
    cout << "Labels: " << endl << labels.transpose() << endl;

    return 0;
}
