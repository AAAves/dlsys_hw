#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void dot(const float *A, const float *B, float *C, int m, int n, int k){
    /*
    impl of A*B, A of size m*n, B of size n*k, C of size m*k
    */
    for (int i=0;i<m;i++){
        for (int l=0;l<k;l++){
            C[i*k+l] = 0.0;
        }
    }
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            for (int l=0;l<k;l++){
                C[i*k+l] += A[i*n+j]*B[j*k+l];
            }
        }
    }

}

void softmax(const float *A, float *B, int m, int n){
    /*
    impl of B = softmax(A), A of size m*n
    */
    float *temp_sum = new float[m];
    
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            B[i*n+j] = exp(A[i*n+j]);
            if (j==0) temp_sum[i]=0.0;
            temp_sum[i]+= B[i*n+j];
        }
    }
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            B[i*n+j] = B[i*n+j]/temp_sum[i];
        }
    }
    delete[] temp_sum;

}

void zero(float *A, int m, int n){
    /*
    init A with zero
    */

    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            A[i*n+j] =0.0;
        }
    }


}

void transpose(const float *A, float *B, int m, int n){
    /*
    impl of B = A.T
    */


    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            B[j*m+i] =A[i*n+j];
        }
    }

}

void substract(const float *A, const float *B, float *C, int m, int n){
    /*
    impl of C = A-B
    */

    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            C[i*n+j] =A[i*n+j] - B[i*n+j];
        }
    }
}


void dotwithnum(const float *A, float *B, int m, int n, float num){
    /*
    impl of B = A * num, A of size m*n
    */
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            B[i*n + j] =  A[i*n + j]*num; 
        }
    }

}

void print_func(const float *A, int m, int n){

    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            std::cout<< A[i*n + j]<<" "; 
        }
            std::cout<<std::endl; 
    }
    std::cout<<std::endl; 
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int iter_cnt = m/batch;
    float *H = new float[batch*k];
    float *Z = new float[batch*k];
    float *I = new float[batch*k];
    float *ZI = new float[batch*k];
    float *XT = new float[batch*n];
    float *G = new float[n*k];
    for(int i=0;i<iter_cnt;i++){
        const float *batch_x = X+i*batch*n;
        const unsigned char *batch_y = y+i*batch;
        dot(batch_x, theta, H, batch, n, k);
        softmax(H,Z,batch,k);
        print_func(Z,batch,k);
        zero(I,batch,k);
        for (int yi=0;yi<batch;yi++){
            I[yi*k+batch_y[yi]] = 1.0;
        }
        transpose(batch_x,XT,batch,n);
        print_func(XT,n,batch);
        substract(Z,I,ZI,batch,k);
        print_func(ZI,batch,k);
        dot(XT, ZI, G, n, batch, k);
        print_func(G,n,k);
        dotwithnum(G,G,n,k,lr/float(batch));
        print_func(G,n,k);
        substract(theta,G,theta,n,k);
        print_func(theta,n,k);
    }
    delete[] H;
    delete[] Z;
    delete[] I;
    delete[] ZI;
    delete[] XT;
    delete[] G;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
