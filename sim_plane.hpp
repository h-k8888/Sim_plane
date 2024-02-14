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
typedef Eigen::Matrix<double, 6, 6> M6D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

extern int refine_maximum_iter;
extern double incident_cov_max, incident_cov_scale, incident_cos_min, bearing_stddev;
extern double range_stddev, lambda_cov_threshold;
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


M3D calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc) {
    float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
    float range_var = range_inc * range_inc;
    float tangent_var = pow(DEG2RAD(degree_inc), 2) * range * range; // d^2 * sigma^2

    Eigen::Vector3d direction(pb);
    direction.normalize();
    M3D rrt = direction * direction.transpose(); // ray * ray^t
    return range_var * rrt + tangent_var * (Eigen::Matrix3d::Identity() - rrt);
}

M3D calcBodyCov(const Eigen::Vector3d &ray, const double & range, const float range_inc, const float degree_inc) {
    float range_var = range_inc * range_inc;
    float tangent_var = pow(DEG2RAD(degree_inc), 2) * range * range; // d^2 * sigma^2

    Eigen::Vector3d direction(ray);
    direction.normalize();
    M3D rrt = direction * direction.transpose(); // ray * ray^t
    return range_var * rrt + tangent_var * (Eigen::Matrix3d::Identity() - rrt);
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

//void printM(const M3D & m, const string & s)
//{
//    cout << s << ":\n" << m <<endl;
//}

template<typename _Scalar, int _Rows, int _Cols>
void printM(const Eigen::Matrix<_Scalar, _Rows, _Cols> & m, const string & s)
{
    cout << s << ":\n" << m <<endl;
}

void calcCloudCov(const vector<V3D>& points, M3D & covariance, V3D & center)
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
    calcCloudCov(cloud, covariance, center);
}

// PCA Self Adjoint Eigen Solver
void PCASelfAdjoint(const M3D & covariance, M3D & eigen_vectors, V3D & eigen_values)
{
    TicToc t_start;
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
}

// PCASelfAdjoint
void PCASelfAdjoint(const vector<V3D>& points, M3D & eigen_vectors, V3D & eigen_values, V3D & center)
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
//    printV(eigen_normal, "PCASelfAdjoint normal");
//    printf("PCASelfAdjoint cost: %.6fms\n", t_cost);
}

void PCASelfAdjoint(const vector<V4D>& points, M3D & eigen_vectors, V3D & eigen_values, V3D & center)
{
    vector<V3D> cloud(points.size());
    for (int i = 0; i < points.size(); ++i) {
        cloud[i] = points[i].head(3);
    }
    PCASelfAdjoint(cloud, eigen_vectors, eigen_values, center);
}

// PCA Eigen Solver
void PCAEigenSolver(const M3D & covariance, M3D & eigen_vectors, V3D & eigen_values)
{
    Eigen::EigenSolver<Eigen::Matrix3d> es(covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    eigen_vectors.col(0) = evecs.real().col(evalsMin);
    eigen_vectors.col(1) = evecs.real().col(evalsMid);
    eigen_vectors.col(2) = evecs.real().col(evalsMax);
    eigen_values(0) = evalsReal[evalsMin];
    eigen_values(1) = evalsReal[evalsMid];
    eigen_values(2) = evalsReal[evalsMax];
}

// PCA Eigen Solver
void EigenSolverNormalCov(const M3D & covariance, const vector<V3D>& points, const V3D & center,
                             M3D & eigen_vectors, V3D & eigen_values, M6D & plane_cov)
{
    TicToc t_start;
    plane_cov = M6D::Zero();
    int points_size = points.size();

    Eigen::EigenSolver<Eigen::Matrix3d> es(covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    eigen_vectors.col(0) = evecMin;
    eigen_vectors.col(1) = evecMid;
    eigen_vectors.col(2) = evecMax;
    eigen_values(0) = evalsReal[evalsMin];
    eigen_values(1) = evalsReal[evalsMid];
    eigen_values(2) = evalsReal[evalsMax];


    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / points_size, 0, 0, 0, 1.0 / points_size, 0, 0, 0,
            1.0 / points_size;
    for (int i = 0; i < points.size(); i++) {
        Eigen::Matrix<double, 6, 3> J;
        Eigen::Matrix3d F = M3D::Zero();
        V3D p_center = points[i] - center;
        F.row(1)  = p_center.transpose() / ((points_size) * (eigen_values(0) - eigen_values(1))) *
                (evecMid * evecMin.transpose() + evecMin * evecMid.transpose());
        F.row(2) = p_center.transpose() / ((points_size) * (eigen_values(0) - eigen_values(2))) *
              (evecMax * evecMin.transpose() + evecMin * evecMax.transpose());

        J.block<3, 3>(0, 0) = eigen_vectors * F;
        J.block<3, 3>(3, 0) = J_Q;

//        plane_cov += J * points[i].cov * J.transpose();
        plane_cov += J * M3D::Identity() * J.transpose();
//            const pointWithCov & pv = points[i];
//            double cos_theta = abs(plane->normal.dot(pv.normal.cast<double>())); // [0, 1.0]
////        double sin_theta2 = 1 - cos_theta * cos_theta;
//            double roughness = (1 - cos_theta) * roughness_cov_scale;
//            M3D point_cov = pv.cov + roughness * M3D::Identity() +
//                            calcIncidentCovScale(pv.ray, pv.p2lidar,  plane->normal) * pv.ray * pv.ray.transpose();
//            plane->plane_cov += J * point_cov * J.transpose();
    }
}

// CovEigenSolverNormalCov, code from VoxelMap
void CovEigenSolverNormalCov(const vector<V3D>& points, M3D & eigen_vectors, V3D & eigen_values, V3D & center,
                             M6D & plane_cov)
{
    TicToc t_start;
    plane_cov = M6D::Zero();
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

    Eigen::EigenSolver<Eigen::Matrix3d> es(covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    eigen_vectors.col(0) = evecMin;
    eigen_vectors.col(1) = evecMid;
    eigen_vectors.col(2) = evecMax;
    eigen_values(0) = evalsReal[evalsMin];
    eigen_values(1) = evalsReal[evalsMid];
    eigen_values(2) = evalsReal[evalsMax];

    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / points_size, 0, 0, 0, 1.0 / points_size, 0, 0, 0,
            1.0 / points_size;
//    if (evalsReal(evalsMin) < 0.1) {

//        plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
//                evecs.real()(2, evalsMin);
//        double t3 = omp_get_wtime();
        std::vector<int> index(points.size());
        std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());
        for (int i = 0; i < points.size(); i++) {
            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F;
            for (int m = 0; m < 3; m++) {
                if (m != (int)evalsMin) {
                    Eigen::Matrix<double, 1, 3> F_m =
                            (points[i] - center).transpose() /
                            ((points_size) * (evalsReal[evalsMin] - evalsReal[m])) *
                            (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                             evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
                    F.row(m) = F_m;
                } else {
                    Eigen::Matrix<double, 1, 3> F_m;
                    F_m << 0, 0, 0;
                    F.row(m) = F_m;
                }
            }
            J.block<3, 3>(0, 0) = evecs.real() * F;
            J.block<3, 3>(3, 0) = J_Q;

//        plane_cov += J * points[i].cov * J.transpose();
        plane_cov += J * M3D::Identity() * J.transpose();
//            const pointWithCov & pv = points[i];
//            double cos_theta = abs(plane->normal.dot(pv.normal.cast<double>())); // [0, 1.0]
////        double sin_theta2 = 1 - cos_theta * cos_theta;
//            double roughness = (1 - cos_theta) * roughness_cov_scale;
//            M3D point_cov = pv.cov + roughness * M3D::Identity() +
//                            calcIncidentCovScale(pv.ray, pv.p2lidar,  plane->normal) * pv.ray * pv.ray.transpose();
//            plane->plane_cov += J * point_cov * J.transpose();
        }

//    double t_cost = t_start.toc();
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


void calcLambdaCov(const vector<M3D> & points_cov, const vector<M3D> & Jpi, vector<double> & lambda_cov)
{
//    assert(points_cov.size() == Jpi.size());
    lambda_cov.resize(3, 0.0);
    for (int i = 0; i < points_cov.size(); ++i) {
        lambda_cov[0] += Jpi[i].row(0) * points_cov[i] * Jpi[i].row(0).transpose();
        lambda_cov[1] += Jpi[i].row(1) * points_cov[i] * Jpi[i].row(1).transpose();
        lambda_cov[2] += Jpi[i].row(2) * points_cov[i] * Jpi[i].row(2).transpose();
    }
}

void calcLambdaCovIncremental(const vector<M3D> & points_cov, const vector<M3D> & Jpi,
                              vector<double> & lambda_cov_old, vector<double> & lambda_cov_incre)
{
//    assert(points_cov.size() == Jpi.size());
    lambda_cov_incre.resize(3);
    double n = (double)points_cov.size();
    double scale = pow((n - 1.0), 2) / (n * n); //^2
    for (int i = 0; i < 3; ++i) {
//        const M3D & point_cov = points_cov.back();
//        const M3D & point_cov = points_cov.back();
        lambda_cov_incre[i] = lambda_cov_old[i] * scale +
                              Jpi.back().row(i) * points_cov.back() * Jpi.back().row(i).transpose();
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
void JacobianLambda(const vector<V3D> & points, const M3D & eigen_vectors,
                    const V3D & center, vector<M3D> & Jpi)
{
    double n = (double)points.size();
    Jpi.resize(points.size());
    vector<M3D> uk_ukt(3);
    for (int i = 0; i < 3; ++i) {
        uk_ukt[i] = 2.0 / n * eigen_vectors.col(i) * eigen_vectors.col(i).transpose(); // 2 / n * v_k * v_k^t
    }
    double cov_lambda1 = 0, cov_lambda2 = 0, cov_lambda3 = 0;
    for (int i = 0; i < points.size(); ++i) {
        V3D center2point = points[i] - center;
        Jpi[i].row(0) = center2point.transpose() * uk_ukt[0];
        Jpi[i].row(1) = center2point.transpose() * uk_ukt[1];
        Jpi[i].row(2) = center2point.transpose() * uk_ukt[2];

        cov_lambda1 += Jpi[i].row(0) * M3D::Identity() * Jpi[i].row(0).transpose();
        cov_lambda2 += Jpi[i].row(1) * M3D::Identity() * Jpi[i].row(1).transpose();
        cov_lambda3 += Jpi[i].row(2) * M3D::Identity() * Jpi[i].row(2).transpose();
    }
}

// specified eigen value
void incrementalDeEigenValue(const vector<V3D> & points, const M3D & eigen_vectors_old, const V3D & center_old,
                             const vector<V3D> & Jpi_old, const M3D & eigen_vectors_new, const int& lambda_i,
                             vector<V3D> & Jpi_new)
{
    double n = (double)points.size();
    const V3D & xn = points.back();
    V3D xn_mn1 = xn - center_old;
    const V3D e1 = eigen_vectors_new.col(lambda_i); // n
    const V3D en_1 = eigen_vectors_new.col(lambda_i); // n - 1

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

// return whether the update is incremental
bool incrementalJacobianLambda(const vector<V3D> & points, const M3D & eigen_vectors_old, const V3D & center_old,
                               const M3D & eigen_vectors_new, const V3D & center_new, vector<M3D> & Jpi_incre)
{
//    assert(points.size() == Jpi_old.size() + 1);
    double n = (double)points.size();
    const V3D & xn = points.back();
    V3D xn_mn1 = xn - center_old;
//    vector<V3D> en(3), en_1(3); // eigen vectors of n, n-1
//    vector<double> cos_en(3), cos_en_1(3), cos_theta(3);
    vector<V3D> term_2(3);
//    printf("derivative increment:\n");
    for (int i = 0; i < 3; ++i) {
        const V3D & en = eigen_vectors_new.col(i); // n
        const V3D & en_1 = eigen_vectors_old.col(i); // n - 1

        double cos_en = abs(xn_mn1.dot(en)); // d * mx_mn1
        double cos_en_1 = abs(xn_mn1.dot(en_1));
        double cos_theta = abs(en.dot(en_1));
        term_2[i] =  ( cos_en * en_1 + cos_en_1 * en_1) / (n * n * cos_theta);
//        printf("lambda increment term %d: %e %e %e\n", i + 1, term_2[i](0), term_2[i](1), term_2[i](2));
    }
//    Jpi_incre.resize(points.size());
    double scale_1 = (n - 1) / n;
    if (term_2[0].norm() > lambda_cov_threshold)
    { //todo
        JacobianLambda(points, eigen_vectors_new, center_new, Jpi_incre);
//        for (int i = 0; i < points.size() - 1; ++i) { // i = 1, 2 ..., n-1
//            Jpi_incre[i].row(0) = scale_1 * Jpi_old[i].row(0) - term_2[0].transpose(); // d lambda1 d p
//            Jpi_incre[i].row(1) = scale_1 * Jpi_old[i].row(1) - term_2[1].transpose(); // d lambda2 d p
//            Jpi_incre[i].row(2) = scale_1 * Jpi_old[i].row(2) - term_2[2].transpose(); // d lambda3 d p
//        }
        return false;
    }
    else {
        Jpi_incre.resize(1);
        // for i = n
        Jpi_incre.back().row(0) = term_2[0].transpose() * (n - 1); // d lambda1 d p
        Jpi_incre.back().row(1) = term_2[1].transpose() * (n - 1); // d lambda2 d p
        Jpi_incre.back().row(2) = term_2[2].transpose() * (n - 1); // d lambda3 d p
        return true;
    }
}

void calcNormalCov(const vector<V3D> & points, const M3D & eigen_vectors, const V3D & eigen_values,
                   const V3D& center, M6D & plane_cov)
{
    const V3D & evecMin = eigen_vectors.col(0);
    const V3D & evecMid = eigen_vectors.col(1);
    const V3D & evecMax = eigen_vectors.col(2);

    int points_size = points.size();
    plane_cov = M6D::Zero();
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / points_size, 0, 0, 0, 1.0 / points_size, 0, 0, 0,
            1.0 / points_size;
    for (int i = 0; i < points.size(); i++) {
        Eigen::Matrix<double, 6, 3> J;
        Eigen::Matrix3d F = M3D::Zero();
        V3D p_center = points[i] - center;
        F.row(1)  = p_center.transpose() / ((points_size) * (eigen_values(0) - eigen_values(1))) *
                    (evecMid * evecMin.transpose() + evecMin * evecMid.transpose());
        F.row(2) = p_center.transpose() / ((points_size) * (eigen_values(0) - eigen_values(2))) *
                   (evecMax * evecMin.transpose() + evecMin * evecMax.transpose());

        J.block<3, 3>(0, 0) = eigen_vectors * F;
        J.block<3, 3>(3, 0) = J_Q;

//        plane_cov += J * points[i].cov * J.transpose();
        plane_cov += J * M3D::Identity() * J.transpose();
//            const pointWithCov & pv = points[i];
//            double cos_theta = abs(plane->normal.dot(pv.normal.cast<double>())); // [0, 1.0]
////        double sin_theta2 = 1 - cos_theta * cos_theta;
//            double roughness = (1 - cos_theta) * roughness_cov_scale;
//            M3D point_cov = pv.cov + roughness * M3D::Identity() +
//                            calcIncidentCovScale(pv.ray, pv.p2lidar,  plane->normal) * pv.ray * pv.ray.transpose();
//            plane->plane_cov += J * point_cov * J.transpose();
    }

}


#endif //SIM_PLANE_SIM_PLANE_H
