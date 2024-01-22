//
// Created by hk on 1/19/24.
//

#ifndef SIM_PLANE_SIM_PLANE_H
#define SIM_PLANE_SIM_PLANE_H

#include <iostream>
#include <math.h>
#include <thread>
#include <fstream>
#include <random>
#include <fstream>
#include <string>
#include <omp.h>
#include <Eigen/Dense>

#include "IKFoM_toolkit/esekfom/esekfom.hpp"

#ifndef DEG2RAD
#define DEG2RAD(x) ((x)*0.017453293)
#endif

//typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
//typedef MTK::S2<double, 98090, 10000, 1> S2;
//typedef MTK::vect<1, double> vect1;
//typedef MTK::vect<2, double> vect2;

using namespace std;
typedef Eigen::Vector3d V3D;
typedef Eigen::Vector4d V4D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;


// 3D point with covariance
typedef struct pointWithCov {
    Eigen::Vector3d point;
    Eigen::Vector3d point_world;
    Eigen::Matrix3d cov;
    Eigen::Matrix3d cov_lidar;
    Eigen::Vector3f normal;
    Eigen::Vector3d ray; // point to lidar(t), a unit vector
    double p2lidar;
    double roughness_cov; // ^2, isotropic cov in 3D space
    double tangent_cov; // ^2, isotropic cov in tangent space
} pointWithCov;

void printV(const V3D & v, const string & s)
{
    cout << s << ":\n" << v.transpose() <<endl;
}

void printM(const M3D & m, const string & s)
{
    cout << s << ":\n" << m <<endl;
}

// PCA
void PCA(const vector<V3D>& points, M3D & eigen_vectors, V3D & eigen_values, V3D & center)
{
//    plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    M3D covariance = Eigen::Matrix3d::Zero();
    center = Eigen::Vector3d::Zero();
    V3D normal = Eigen::Vector3d::Zero();

    int points_size = points.size();

    for (int i = 0; i < points_size; ++i) {
        const V3D& pv = points[i];
        covariance += pv * pv.transpose();
        center += pv;
    }
    center = center / points_size;
    covariance = covariance / points_size - center * center.transpose();

    Eigen::SelfAdjointEigenSolver<M3D> saes(covariance);
    eigen_vectors = saes.eigenvectors();
    eigen_values = saes.eigenvalues(); // lambda_0 < lambda_1 < lambda_2
    const double evals_min = eigen_values(0);
    const double evals_mid = eigen_values(1);
    const double evals_max = eigen_values(2);
    V3D eigen_normal = eigen_vectors.col(0);
    V3D eigen_v_mid = eigen_vectors.col(1);
    V3D eigen_v_max = eigen_vectors.col(2);
    printV(eigen_normal, "PCA normal");
}

void PCA(const vector<V4D>& points, M3D & eigen_vectors, V3D & eigen_values, V3D & center)
{
    vector<V3D> cloud(points.size());
    for (int i = 0; i < points.size(); ++i) {
        cloud[i] = points[i].head(3);
    }
    PCA(cloud, eigen_vectors, eigen_values, center);
}


void saveCloud(const vector<V4D> & cloud, string& file)
{
//    printf("cloud file: %s\n", file.c_str());
    ofstream of(file);
    if (of.is_open())
    {
        of.setf(ios::fixed, ios::floatfield);
        of.precision(6);
        for (int i = 0; i < (int)cloud.size(); ++i) {
            of<< cloud[i](0) << " " << cloud[i](1) << " " << cloud[i](2) << " " << cloud[i](3) << endl;
        }
        of.close();
    }
}

void saveCloud(const vector<vector<V4D>> & cloud_per_lidar, string& file)
{
    for (int i = 0; i < (int)cloud_per_lidar.size(); ++i) {
        string file_name =  file + "_" + to_string(i) + ".txt";
        saveCloud(cloud_per_lidar[i], file_name);
    }
}

double diff_normal(const V3D & n1, const V3D & n2)
{
    V3D x_cross_y = n1.cross(n2);
    double x_dot_y = n1.dot(n2);
    double theta = atan2(x_cross_y.norm(), abs(x_dot_y));

//    Eigen::Matrix<double, 3, 2> B;
//    B.col(0) = eigen_v_mid;
//    B.col(1) = eigen_v_max;
//
//    Eigen::Vector2d u = B.transpose() * x_cross_y / x_cross_y.norm() * theta;
//    V3D Bu = B * u;
//    M3D R_Bu = SO3::exp(Bu).toRotationMatrix();
//
//    V3D normal_rotate = R_Bu * n1;
//
//    eigen_v_mid = R_Bu * eigen_v_mid;
//    eigen_v_max = R_Bu * eigen_v_max;
////    plane->plane_cov.block<3, 3>(0, 0) = R_Bu * plane->plane_cov.block<3, 3>(0, 0) * R_Bu.transpose();
    return theta;
}


double point2planeResidual(const vector<V4D> & points, const V3D & centroid, const V3D & normal)
{
    double sum_dist2 = 0;
    for (int i = 0; i < points.size(); ++i) {
        double dist = normal.dot(points[i].head(3) - centroid);
        sum_dist2 += dist * dist;
    }
    return sum_dist2;
}


#endif //SIM_PLANE_SIM_PLANE_H
