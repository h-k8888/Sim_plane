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

#include "tic_toc.h"

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

extern int refine_maximum_iter;
extern double incident_cov_max, incident_cov_scale, incident_cos_min, bearing_stddev;
extern double range_stddev;
// ray: unit vector
double calcIncidentCovScale(const V3D & ray, const double & dist, const V3D& normal)
{
//    static double angle_rad = DEG2RAD(angle_cov);
    double cos_incident = abs(normal.dot(ray));
    cos_incident = max(cos_incident, incident_cos_min);
    double sin_incident = sqrt(1 - cos_incident * cos_incident);
    double sigma_a = dist * sin_incident / cos_incident * bearing_stddev; // range * tan(incident) * sigma_angle
    return min(incident_cov_max, incident_cov_scale * sigma_a * sigma_a); // scale * sigma_a^2
}

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

    void getCovValues(double & range_cov, double & tan_cov, const V3D & n) const
    {
        double incident_scale = calcIncidentCovScale(ray, p2lidar, n);
        range_cov = range_stddev * range_stddev + roughness_cov + incident_scale;
        tan_cov = tangent_cov;
    }
} pointWithCov;

void printV(const V3D & v, const string & s)
{
    cout << s << ":\n" << v.transpose() <<endl;
}

void printM(const M3D & m, const string & s)
{
    cout << s << ":\n" << m <<endl;
}

void calcCovMatrix(const vector<V3D>& points, M3D & covariance, V3D & center)
{
    int points_size = points.size();
    center = Eigen::Vector3d::Zero();
    V3D normal = Eigen::Vector3d::Zero();

    for (int i = 0; i < points_size; ++i) {
        const V3D& pv = points[i];
        covariance += pv * pv.transpose();
        center += pv;
    }
    center = center / points_size;
    covariance = covariance / points_size - center * center.transpose();
}


void calcCovMatrix(const vector<V4D>& points, M3D & covariance, V3D & center)
{
    vector<V3D> cloud(points.size());
    for (int i = 0; i < points.size(); ++i) {
        cloud[i] = points[i].head(3);
    }
    calcCovMatrix(cloud, covariance, center);
}


// PCA
void PCA(const vector<V3D>& points, M3D & eigen_vectors, V3D & eigen_values, V3D & center)
{
    TicToc t_start;
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

    double t_cost = t_start.toc();
//    printV(eigen_normal, "PCA normal");
//    printf("PCA cost: %.6fms\n", t_cost);

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

double point2planeResidual(const vector<pointWithCov> & points, const V3D & centroid, const V3D & normal)
{
    double sum_dist2 = 0;
    for (int i = 0; i < points.size(); ++i) {
        double dist = normal.dot(points[i].point - centroid);
        sum_dist2 += dist * dist;
    }
    return sum_dist2;
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


// dir must be a unit vector
void findLocalTangentBases(const V3D& dir, V3D & base1, V3D & base2)
{
    TicToc t_start;
    // find base vector in the local tangent space
    base1 = dir.cross(V3D(1.0, 0, 0));
    if (dir(0) == 1.0)
        base1 = dir.cross(V3D(0, 0, 1.0));
    base1.normalize();
    base2 = dir.cross(base1);
    base2.normalize();
//    printf("findLocalTangentBases cost: %fms\n", t_start.toc());
}

double angle_threshold = 0.1;
double dist_threshold = 0.001;
void refineNormalAndCenterDWithCov(const V3D& normal_input, const std::vector<pointWithCov> &points,
                                   const V3D & center_input, V3D & normal_out, V3D & center_out)
{
    bool is_converge = true;
    V3D normal = normal_input;
    V3D center = center_input;
    int i = -1;
    for(; i < refine_maximum_iter; ++i)
    {
//        double resudial_merged_refine = point2planeResidual(points, center, normal);
//        double theta_merged_refine = diff_normal(normal_input, normal);
//        printf("refined iter %d\nnormal diff: %f deg\nsum residual^2: %f\n\n", i, theta_merged_refine / M_PI * 180.0,
//               resudial_merged_refine);

        // find base vector in the local tangent space
        V3D bn1, bn2;
        findLocalTangentBases(normal, bn1, bn2);

        M3D NNt = normal * normal.transpose();
        M3D JtJ = M3D::Zero();
        V3D Jte = V3D::Zero();
        for (int j = 0; j < points.size(); j++) {
            const pointWithCov &pv = points[j];
            // point to plane E dist^2
            V3D c2p = pv.point - center;

            double range_cov, tan_cov;
            pv.getCovValues(range_cov, tan_cov, normal);
            double range_var_inv = 1.0 / sqrt(range_cov); // 1 / sigma range
            double tangent_var_inv = 1.0 / sqrt(tan_cov); // 1 / (range * sigma angle)

            // construct V
            const V3D &ray = pv.ray;
            V3D br1, br2;
            findLocalTangentBases(ray, br1, br2);
            M3D A;
            A.row(0) = range_var_inv * ray.transpose();
            A.row(1) = tangent_var_inv * br1.transpose();
            A.row(2) = tangent_var_inv * br2.transpose();
//            A.row(0) = ray.transpose();
//            A.row(1) = br1.transpose();
//            A.row(2) = br2.transpose();
            M3D Jw_i;
//            double NtP = normal.transpose() * c2p;
            Jw_i.col(0) = A * (bn1 * normal.transpose() + normal * bn1.transpose()) * c2p;
            Jw_i.col(1) = A * (bn2 * normal.transpose() + normal * bn2.transpose()) * c2p;
            Jw_i.col(2) = -A * normal;
            V3D e_i = A * NNt * c2p;
            JtJ += Jw_i.transpose() * Jw_i;

            V3D Jte_i = Jw_i.transpose() * e_i;
            Jte += Jte_i;
//                cout << "Jw_i\n" << Jw_i << endl;
//                cout << "Jte_i\n" << Jte_i.transpose() << endl;
        }
//            cout << "JtJ:\n" << JtJ << endl;
//            cout << "JtJ inv:\n" << JtJ.inverse() << endl;
//            cout << "Jte:\n" << Jte.transpose() << endl;

        V3D d_w = -(JtJ).inverse() * Jte;
//            ROS_INFO("delta_w [%f %f] delta_d [%f]", d_w(0), d_w(1), d_w(2));

        V3D dn = d_w(0) * bn1 + d_w(1) * bn2; // diff normal
        V3D normal_refine = normal + dn;
        normal_refine.normalize();
        center += normal_refine * d_w(2);

        double resudial_merged_refine = point2planeResidual(points, center, normal_refine);
        double theta_merged_refine = diff_normal(normal, normal_refine);
        double diff_angle = theta_merged_refine / M_PI * 180.0;

        printf("\nrefined iter %d\nnormal diff: %f deg\ndist: %f\nsum residual^2: %f\n",
               i, diff_angle, d_w(2), resudial_merged_refine);
        cout << "Jte: " << Jte.transpose() << endl;


        normal = normal_refine;

//        for (int j = 0; j < 3; ++j)
//        {
            if (diff_angle < angle_threshold && d_w(2) < dist_threshold) {
//                is_converge = false;
                break;
            }
//        }

//        if (is_converge)
//            break;
    }
//        ROS_INFO("%d iter normal old [%f %f %f] new [%f %f %f]", i, normal_input(0), normal_input(1), normal_input(2),
//                 normal(0), normal(1), normal(2));
//        ROS_INFO("center old [%f %f %f] new [%f %f %f]", plane->center(0), plane->center(1), plane->center(2),
//                 center(0), center(1), center(2));
    normal_out = normal;
    center_out = center;
}

V3D refineNormalWithCov(const V3D& normal_input, const std::vector<pointWithCov> &points, const V3D & center_input)
{
//    bool is_converge = true;
    V3D normal = normal_input;
    int i = -1;
    for(; i < refine_maximum_iter; ++i)
    {
        // find base vector in the local tangent space
        V3D bn1, bn2;
        findLocalTangentBases(normal, bn1, bn2);

        M3D NNt = normal * normal.transpose();
        Eigen::Matrix2d JtJ = Eigen::Matrix2d::Zero();
        Eigen::Vector2d Jte = Eigen::Vector2d::Zero();
        for (int j = 0; j < points.size(); j++) {
            const pointWithCov &pv = points[j];
            // point to plane E dist^2
            V3D c2p = pv.point - center_input;

            double range_cov, tan_cov;
            pv.getCovValues(range_cov, tan_cov, normal);
            double range_var_inv = 1.0 / sqrt(range_cov); // 1 / sigma range
            double tangent_var_inv = 1.0 / sqrt(tan_cov); // 1 / (range * sigma angle)

            // construct V
            // todo record br1, br2
            V3D ray = pv.ray;
            ray.normalize();
            V3D br1, br2;
            findLocalTangentBases(ray, br1, br2);
            M3D A;
            A.row(0) = range_var_inv * ray.transpose();
            A.row(1) = tangent_var_inv * br1.transpose();
            A.row(2) = tangent_var_inv * br2.transpose();
//            A.row(0) = ray.transpose();
//            A.row(1) = br1.transpose();
//            A.row(2) = br2.transpose();

//            cout << "r dot b1: " << ray.dot(br1) << endl;
//            cout << "r dot b2: " << ray.dot(br2) << endl;
//            assert(ray.dot(br1) == 0);
//            assert(ray.dot(br2) == 0);
//            cout << "AtA_i:\n" << A.transpose() * A << endl;

            Eigen::Matrix<double, 3, 2> Jw_i;
//            double NtP = normal.transpose() * c2p;
            Jw_i.col(0) = A * (bn1 * normal.transpose() + normal * bn1.transpose()) * c2p;
            Jw_i.col(1) = A * (bn2 * normal.transpose() + normal * bn2.transpose()) * c2p;
            V3D e_i = A * NNt * c2p;
//            cout << "JtJ_i:\n" << Jw_i.transpose() * Jw_i << endl;
            JtJ += Jw_i.transpose() * Jw_i;
            Eigen::Vector2d Jte_i = Jw_i.transpose() * e_i;
            Jte += Jte_i;
//            cout << "Jw_i" << Jw_i << endl;
        }
//            cout << "JtJ:\n" << JtJ << endl;
//            cout << "Jte:\n" << Jte.transpose() << endl;

        Eigen::Vector2d d_w = -(JtJ).inverse() * Jte;
        V3D dn = d_w(0) * bn1 + d_w(1) * bn2;

        V3D normal_refine = normal + dn;
        normal_refine.normalize();

        double resudial_merged_refine = point2planeResidual(points, center_input, normal_refine);
        double theta_merged_refine = diff_normal(normal, normal_refine);
        double diff_angle = theta_merged_refine / M_PI * 180.0;
        printf("\nrefined iter %d\nnormal diff: %f deg\nsum residual^2: %f\n",
               i, diff_angle, resudial_merged_refine);
        cout << "Jte: " << Jte.transpose() << endl;

        normal = normal_refine;
        if (diff_angle < angle_threshold)
            break;
//            is_converge = false;
//
////            ROS_INFO("dw [%f %f]  dn [%f %f %f]", d_w(0), d_w(1), dn(0), dn(1), dn(2));
//
//        if (is_converge)
//            break;
    }
//        ROS_INFO("%d iter normal old [%f %f %f] new [%f %f %f]", i, normal_input(0), normal_input(1), normal_input(2),
//                 normal(0), normal(1), normal(2));
    return normal;
}

V3D refineNormal(const V3D& normal_input, const std::vector<pointWithCov> &points, const V3D & center_input)
{
    V3D normal = normal_input;
    int i = -1;
    for(; i < refine_maximum_iter; ++i)
    {
        TicToc t_start;

        // find base vector in the local tangent space
        V3D bn1, bn2;
        findLocalTangentBases(normal, bn1, bn2);
        double cost_base = t_start.toc();

        M3D NNt = normal * normal.transpose();
        Eigen::Matrix2d JtJ = Eigen::Matrix2d::Zero();
        Eigen::Vector2d Jte = Eigen::Vector2d::Zero();
        for (int j = 0; j < points.size(); j++) {
            const pointWithCov &pv = points[j];
            // point to plane E dist^2
            V3D c2p = pv.point - center_input;

            // construct V
            V3D ray = pv.ray;
            M3D A;

//            ray.normalize();
            V3D br1, br2;
            findLocalTangentBases(ray, br1, br2);
            A.row(0) = ray.transpose();
            A.row(1) = br1.transpose();
            A.row(2) = br2.transpose();
//            A.row(0) = ray.transpose();
//            V3D br1 = normal.cross(ray).normalized();
//            A.row(1) = br1; // br1
//            A.row(2) = ray.cross(br1).normalized(); //br2

//            cout << "r dot b1: " << ray.dot(br1) << endl;
//            cout << "r dot b2: " << ray.dot(br2) << endl;
//            assert(ray.dot(br1) == 0);
//            assert(ray.dot(br2) == 0);
//            cout << "AtA_i:\n" << A.transpose() * A << endl;

            Eigen::Matrix<double, 3, 2> Jw_i;
//            double NtP = normal.transpose() * c2p;
            Jw_i.col(0) = A * (bn1 * normal.transpose() + normal * bn1.transpose()) * c2p;
            Jw_i.col(1) = A * (bn2 * normal.transpose() + normal * bn2.transpose()) * c2p;
            V3D e_i = A * NNt * c2p;
//            cout << "JtJ_i:\n" << Jw_i.transpose() * Jw_i << endl;
            JtJ += Jw_i.transpose() * Jw_i;
            Eigen::Vector2d Jte_i = Jw_i.transpose() * e_i;
            Jte += Jte_i;
//            cout << "Jw_i" << Jw_i << endl;
        }
//            cout << "JtJ:\n" << JtJ << endl;
//            cout << "Jte:\n" << Jte.transpose() << endl;

        Eigen::Vector2d d_w = -(JtJ).inverse() * Jte;

        double t_cost = t_start.toc();

        V3D dn = d_w(0) * bn1 + d_w(1) * bn2;

        V3D normal_refine = normal + dn;
        normal_refine.normalize();

        double resudial_merged_refine = point2planeResidual(points, center_input, normal_refine);
        double theta_merged_refine = diff_normal(normal, normal_refine);
        double diff_angle = theta_merged_refine / M_PI * 180.0;
//        printf("\nrefined iter %d\nnormal diff: %f deg\nsum residual^2: %f\n",
//               i, diff_angle, resudial_merged_refine);
//        cout << "Jte: " << Jte.transpose() << endl;

        normal = normal_refine;

        printf("find local bases cost: %fms\nrefine iter once cost: %.6fms\n", cost_base, t_cost);

        if (diff_angle < angle_threshold)
            break;
    }
//        ROS_INFO("%d iter normal old [%f %f %f] new [%f %f %f]", i, normal_input(0), normal_input(1), normal_input(2),
//                 normal(0), normal(1), normal(2));
    return normal;
}

V3D fasterRefineNormal(const V3D& normal_input, const std::vector<pointWithCov> &points, const V3D & center_input)
{
    V3D normal = normal_input;
    int i = -1;
    for(; i < refine_maximum_iter; ++i)
    {
        TicToc t_start;

        // find base vector in the local tangent space
        V3D bn1, bn2;
        findLocalTangentBases(normal, bn1, bn2);
        double cost_base = t_start.toc();

        M3D NNt = normal * normal.transpose();
        Eigen::Matrix2d JtJ = Eigen::Matrix2d::Zero();
        Eigen::Vector2d Jte = Eigen::Vector2d::Zero();
        for (int j = 0; j < points.size(); j++) {
            const pointWithCov &pv = points[j];
            // point to plane E dist^2
            V3D c2p = pv.point - center_input;

            // construct V
            V3D ray = pv.ray;
            M3D A;

//            ray.normalize();
//            V3D br1, br2;
//            findLocalTangentBases(ray, br1, br2);
//            A.row(0) = ray.transpose();
//            A.row(1) = br1.transpose();
//            A.row(2) = br2.transpose();
            A.row(0) = ray.transpose();
            V3D br1 = normal.cross(ray).normalized();
            A.row(1) = br1; // br1
            A.row(2) = ray.cross(br1).normalized(); //br2

            Eigen::Matrix<double, 3, 2> Jw_i;
//            double NtP = normal.transpose() * c2p;
            Jw_i.col(0) = A * (bn1 * normal.transpose() + normal * bn1.transpose()) * c2p;
            Jw_i.col(1) = A * (bn2 * normal.transpose() + normal * bn2.transpose()) * c2p;
            V3D e_i = A * NNt * c2p;
//            cout << "JtJ_i:\n" << Jw_i.transpose() * Jw_i << endl;
            JtJ += Jw_i.transpose() * Jw_i;
            Eigen::Vector2d Jte_i = Jw_i.transpose() * e_i;
            Jte += Jte_i;
//            cout << "Jw_i" << Jw_i << endl;
        }
//            cout << "JtJ:\n" << JtJ << endl;
//            cout << "Jte:\n" << Jte.transpose() << endl;

        Eigen::Vector2d d_w = -(JtJ).inverse() * Jte;

        double t_cost = t_start.toc();

        V3D dn = d_w(0) * bn1 + d_w(1) * bn2;

        V3D normal_refine = normal + dn;
        normal_refine.normalize();

        double resudial_merged_refine = point2planeResidual(points, center_input, normal_refine);
        double theta_merged_refine = diff_normal(normal, normal_refine);
        double diff_angle = theta_merged_refine / M_PI * 180.0;
//        printf("\nrefined iter %d\nnormal diff: %f deg\nsum residual^2: %f\n",
//               i, diff_angle, resudial_merged_refine);
//        cout << "Jte: " << Jte.transpose() << endl;

        normal = normal_refine;

        printf("find local bases cost: %fms\nrefine iter once cost: %.6fms\n", cost_base, t_cost);

        if (diff_angle < angle_threshold)
            break;
    }
//        ROS_INFO("%d iter normal old [%f %f %f] new [%f %f %f]", i, normal_input(0), normal_input(1), normal_input(2),
//                 normal(0), normal(1), normal(2));
    return normal;
}

void printIncidentCov()
{
    double dist = 5;
    for (int i = 0; i <= 90; ++i) {
        double incident = (double)i / 180.0 * M_PI;
        double sigma_incident = dist * tan(incident) * bearing_stddev; // range * tan(incident) * sigma_angle
        double cov_incident = sigma_incident * sigma_incident;
        double cov_inv_n = cos(incident) * cos(incident) / (range_stddev * range_stddev + cov_incident) +
                sin(incident) *  sin(incident) / (bearing_stddev * bearing_stddev );
        double cov_ra_n = cos(incident) * cos(incident) / (range_stddev * range_stddev) +
                           sin(incident) * sin(incident) / (bearing_stddev * bearing_stddev);
        printf("%i degree cov inv:\nincident %f\talong normal %f(inc)\t%f(w\\o inc)\n", i, cov_incident, cov_inv_n, cov_ra_n);
    }
}

/// partial derivative of eigen value with respect to point
void derivativeEigenValue(const vector<V3D> & points, const M3D & eigen_vectors,
                          const V3D & center, const int& lambda_i, vector<V3D> & Jpi)
{
    double n = (double)points.size();
//    V3D point_center = V3D::Zero();
    M3D tmp = 2.0 / n * eigen_vectors.col(lambda_i) * eigen_vectors.col(lambda_i).transpose(); // 2 / n * v_k * v_k^t
    Jpi.resize(points.size());
    for (int i = 0; i < points.size(); ++i) {
        Jpi[i] = (points[i] - center).transpose() * tmp;
    }

}

//
void incrementalDeEigenValue(const vector<V3D> & points, const M3D & eigen_vectors_old, const V3D & center_old,
                             const vector<V3D> & Jpi_old, const M3D & eigen_vectors_new, vector<V3D> & Jpi_new)
{
    double n = (double)points.size();
    const V3D & xn = points.back();
    V3D xn_mn1 = xn - center_old;
    const V3D e1 = eigen_vectors_new.col(0); // n
//    const V3D e2 = eigen_vectors_new.col(1);
//    const V3D e3 = eigen_vectors_new.col(2);
    const V3D en_1 = eigen_vectors_new.col(0); // n - 1
//    const V3D en_2 = eigen_vectors_new.col(1);
//    const V3D en_3 = eigen_vectors_new.col(2);
//    vector<double> cos_vectors(3); // en * en_1
//    for (int i = 0; i < 3; ++i)
//        cos_vectors[i] = abs(eigen_vectors_new.col(i).dot(eigen_vectors_old.col(i)));

    double cos_n = abs(xn_mn1.dot(e1));
    double cos_n_1 = abs(xn_mn1.dot(en_1));
    double cos_theta = abs(e1.dot(en_1));
    Jpi_new.resize(points.size());
    double scale_1 = (n - 1) / n;
    V3D term_2 =  (cos_n * en_1 + cos_n_1 * e1) / (n * n * cos_theta);
    for (int i = 0; i < points.size() - 1; ++i) { // i = 1, 2 ..., n-1
        Jpi_new[i] = scale_1 * Jpi_old[i] - term_2;
    }

    // for i = n
    Jpi_new.back() = term_2 * (n - 1);
}

#endif //SIM_PLANE_SIM_PLANE_H
