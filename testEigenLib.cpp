//#include <iostream>
//#include <Eigen/Core>

///**
// * Apply exponential function for each element in matrix a
// */
//MatrixXd compute_exp(MatrixXd a) {
//    int m = a.rows(), n = a.cols();
//    MatrixXd exp_a(m, n);

//    for(int i = 0; i < m; ++i) {
//        for(int j = 0; j < n; ++j) {
//            exp_a(i, j) = safe_exp( a(i,j) );
////            cout << exp_a(i, j) << " ";
//        }
////        cout << "\n";
//    }

//    return exp_a;
//}

//int test() {
//    MatrixXd exp_a, a;
//    // apply softmax function to compute y_k at each timestep t
//    exp_a = compute_exp(a);
//    eSum = exp_a.rowwise().sum(); //compute the sum of the elements of each row
//    for(int t = 0; t < T; ++t) {
//        y.row(t) = exp_a.row(t)/eSum[t];
//    //        for(int k = 0; k < K; ++k)
//    //            cout << y(t, k) << " ";
//    //        cout << "\n";
//    }

//    return 0;
//}

