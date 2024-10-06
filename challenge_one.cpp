#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <fstream>
#include <lis.h>

using namespace Eigen;
using namespace cv;
using namespace std;

// funzione usata per la creazione del kernel di convoluzione A. richiede come argomenti il numero di righe, il numero di colonne e la matrice del kernel
SparseMatrix<double> createConvolutionMatrix(int rows, int cols, const Matrix3d& kernel) {
    int totalSize = rows * cols;
    SparseMatrix<double> A1(totalSize, totalSize);
    std::vector<Triplet<double>> tripletList;

    // Creazione della matrice A di convoluzione
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;

            // Applicazione del kernel
            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int neighborRow = i + ki;
                    int neighborCol = j + kj;

                    // Check that neighbors are within the image
                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        int neighborIndex = neighborRow * cols + neighborCol;
                        tripletList.push_back(Triplet<double>(index, neighborIndex, kernel(ki + 1, kj + 1)));
                    }
                }
            }
        }
    }

    A1.setFromTriplets(tripletList.begin(), tripletList.end());
    return A1;
}

// Funzione per l'applicazione della convoluzione e la conversione in matrice. Richiede la matrice di convoluzione A, il vettore v e le dimensioni della matrice
MatrixXd applyConvolutionAndConvert(const SparseMatrix<double>& A, const VectorXd& v, int rows, int cols) {
    // Applicazione della convoluzione
    VectorXd smoothedImageVector = A * v;

    // Riporta a matrice
    MatrixXd smoothedImage(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            // Ensure the value is between 0 and 255
            smoothedImage(i, j) = std::max(0.0, std::min(255.0, smoothedImageVector(index)));
        }
    }
    return smoothedImage;
}

// Funzione di debug che riporta i dati statistici della matrice
void printMatrixStatistics(const SparseMatrix<double>& A) {
    int nonZeroCount = A.nonZeros();
    long long totalElements = static_cast<long long>(A.rows()) * A.cols();
    double sparsity = (1.0 - static_cast<double>(nonZeroCount) / totalElements) * 100.0;

    cout << "Numero di elementi non zero: " << nonZeroCount << endl;
    cout << "Numero totale di elementi: " << totalElements << endl;
    cout << "Sparsità: " << sparsity << "%" << endl;
}


int main(int argc, char* argv[]) {
    //Caricamento dell'immagine
    Mat img = imread("/home/karim/Scrivania/NLA_Challenges/first_challenge/128px-Albert_Einstein_Head.jpeg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Errore nel caricamento dell'immagine!" << endl;
        return -1;
    }

    int rows = img.rows;
    int cols = img.cols;
    cout << "Size of the matrix: " << rows << " x " << cols << endl;

    //Conversione dell'immagine in una matrice di Eigen
    MatrixXd imgMatrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            imgMatrix(i, j) = img.at<uchar>(i, j);
        }
    }

    // Introduzione del disturbo nell'immagine
    Mat noisyImg = img.clone();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-50, 50);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            noisyImg.at<uchar>(i, j) = saturate_cast<uchar>(noisyImg.at<uchar>(i, j) + dis(gen));
        }
    }

    // Salvataggio dell'immagine rumorosa nel file noisy_image.png
    imwrite("noisy_image.png", noisyImg);

    // Riformula l'immagine originale come vettore v
    VectorXd v = VectorXd::Zero(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            v(i * cols + j) = img.at<uchar>(i, j);
        }
    }

    // Riformula l'immagine rumorosa come vettore w
    VectorXd w = VectorXd::Zero(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            w(i * cols + j) = noisyImg.at<uchar>(i, j);
        }
    }

    // Verifica delle dimensioni dei vettori
    std::cout << "Size of vector v: " << v.size() << std::endl;
    std::cout << "Size of vector w: " << w.size() << std::endl;

    // Calcolo della norma euclidea del vettore v
    double euclideanNormV = v.norm();
    std::cout << "Euclidean norm of vector v: " << euclideanNormV << std::endl;

    // Definizione del kernel di smoothing 3x3 H_av2
    Matrix3d smoothingKernel;
    smoothingKernel << 1, 1, 1,
                       1, 1, 1,
                       1, 1, 1;
    smoothingKernel /= 9.0;

    // Creazione della matrice di convoluzione A1
    SparseMatrix<double> A1 = createConvolutionMatrix(rows, cols, smoothingKernel);

    // Stampa le statistiche della matrice A1
    printMatrixStatistics(A1);

    // Applicazione della convoluzione e conversione in matrice sulla immagine rumorosa A1 * w
    MatrixXd smoothedImageMatrix = applyConvolutionAndConvert(A1, w, rows, cols);

    // Conversione della matrice Eigen ottenuta dalla applicazione della convoluzione A1 in OpenCV Mat
    Mat smoothedImage(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            smoothedImage.at<uchar>(i, j) = static_cast<uchar>(smoothedImageMatrix(i, j));
        }
    }

    // Salvataggio della immagine smoothed
    imwrite("smoothed_image.jpeg", smoothedImage);

    // Definizione del kernel di sharpening 3x3 H_sh2
    Matrix3d sharpeningKernel;
    sharpeningKernel << 0, -3, 0,
                        -1, 9, -3,
                        0, -1, 0;

    // Creazione della matrice di convoluzione A2
    SparseMatrix<double> A2 = createConvolutionMatrix(rows, cols, sharpeningKernel);

    // Stampa le statistiche della matrice A2
    printMatrixStatistics(A2);

    // Controlla se la matrice A2 è simmetrica
    bool isSymmetricA2 = A2.isApprox(A2.transpose());
    std::cout << "Is A2 symmetric? " << (isSymmetricA2 ? "Yes" : "No") << std::endl;

    // Applicazione del filtro di sharpening all'immagine originale
    MatrixXd sharpenedImageMatrix = applyConvolutionAndConvert(A2, v, rows, cols);

    // Conversione della matrice Eigen ottenuta dalla applicazione della convoluzione A2 in OpenCV Mat
    Mat sharpenedImage(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            sharpenedImage.at<uchar>(i, j) = static_cast<uchar>(sharpenedImageMatrix(i, j));
        }
    }

    // Salvataggio dell'immagine sharpened
    imwrite("sharpened_image.jpeg", sharpenedImage);

    // Creazione del kernel di edge detection 3x3 H_lap
    Matrix3d laplacianKernel;
    laplacianKernel << 0, -1, 0,
                       -1, 4, -1,
                       0, -1, 0;

    // Creazione della matrice di convoluzione A3
    SparseMatrix<double> A3 = createConvolutionMatrix(rows, cols, laplacianKernel);

    // Stampa le statistiche della matrice A3
    printMatrixStatistics(A3);

    // Controlla se la matrice A3 è simmetrica
    bool isSymmetricA3 = A3.isApprox(A3.transpose());
    std::cout << "Is A3 symmetric? " << (isSymmetricA3 ? "Yes" : "No") << std::endl;

    // Applicazione del filtro di edge detection all'immagine originale
    MatrixXd edgeDetectedImageMatrix = applyConvolutionAndConvert(A3, v, rows, cols);

    // Conversione della matrice Eigen ottenuta dalla applicazione della convoluzione A3 in OpenCV Mat
    Mat edgeDetectedImage(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            edgeDetectedImage.at<uchar>(i, j) = static_cast<uchar>(edgeDetectedImageMatrix(i, j));
        }
    }

    // Salvataggio dell'immagine edge detected
    imwrite("edge_detected_image.jpeg", edgeDetectedImage);

    // Create identity matrix I
    /*SparseMatrix<double> I(rows * cols, rows * cols);
    I.setIdentity();

    // Form the matrix I + A3
    SparseMatrix<double> I_plus_A3 = I + A3;

    // Use an iterative solver to solve (I + A3)y = w
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver;
    solver.setTolerance(1e-10);
    solver.compute(I_plus_A3);
    VectorXd y = solver.solve(w);

    // Report iteration count and final residual
    std::cout << "Iteration count: " << solver.iterations() << std::endl;
    std::cout << "Final residual: " << solver.error() << std::endl;

    // Convert the solution vector y back to an image
    MatrixXd yMatrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            yMatrix(i, j) = std::max(0.0, std::min(255.0, y(index)));
        }
    }

    // Convert Eigen matrix back to OpenCV Mat
    Mat yImage(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            yImage.at<uchar>(i, j) = static_cast<uchar>(yMatrix(i, j));
        }
    }

    // Save the image
    imwrite("solution_image.png", yImage);*/

    return 0;
}