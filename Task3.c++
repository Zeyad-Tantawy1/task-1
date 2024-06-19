#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ml;

const int IMG_SIZE = 64; // Resize images to 64x64

// Function to load and preprocess images
void loadImagesFromFolder(const string& folder, int label, vector<Mat>& images, vector<int>& labels) {
    vector<string> filenames;
    glob(folder, filenames);
    for (auto& filename : filenames) {
        Mat img = imread(filename, IMREAD_GRAYSCALE);
        if (!img.empty()) {
            resize(img, img, Size(IMG_SIZE, IMG_SIZE));
            images.push_back(img);
            labels.push_back(label);
        }
    }
}

// Function to convert images to feature vectors
Mat convertImagesToMatrix(const vector<Mat>& images) {
    Mat data(static_cast<int>(images.size()), IMG_SIZE * IMG_SIZE, CV_32FC1);
    for (size_t i = 0; i < images.size(); ++i) {
        Mat img_row = images[i].reshape(1, 1); // Flatten the image to a single row
        img_row.convertTo(data.row(static_cast<int>(i)), CV_32FC1, 1.0 / 255.0); // Normalize
    }
    return data;
}

int main() {
    string catDir = "path_to_your_dataset/cats/*";
    string dogDir = "path_to_your_dataset/dogs/*";

    // Load and preprocess images
    vector<Mat> images;
    vector<int> labels;
    loadImagesFromFolder(catDir, 0, images, labels); // Label for cats is 0
    loadImagesFromFolder(dogDir, 1, images, labels); // Label for dogs is 1

    // Convert images to feature matrix
    Mat data = convertImagesToMatrix(images);

    // Split the dataset into training and testing sets (80-20 split)
    int trainSize = static_cast<int>(0.8 * data.rows);
    Mat trainData = data.rowRange(0, trainSize);
    Mat testData = data.rowRange(trainSize, data.rows);
    Mat trainLabels = Mat(labels).rowRange(0, trainSize);
    Mat testLabels = Mat(labels).rowRange(trainSize, data.rows);

    // Initialize and train the SVM model
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setC(1.0);
    svm->train(trainData, ROW_SAMPLE, trainLabels);

    // Predict using the trained model
    Mat predictedLabels;
    svm->predict(testData, predictedLabels);

    // Evaluate the model
    int correctPredictions = 0;
    for (int i = 0; i < testData.rows; ++i) {
        if (predictedLabels.at<float>(i, 0) == testLabels.at<int>(i, 0)) {
            correctPredictions++;
        }
    }
    float accuracy = static_cast<float>(correctPredictions) / testData.rows;
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;

    return 0;
}
