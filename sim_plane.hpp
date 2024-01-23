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


// dir must be a unit vector
void findLocalTangentBases(const V3D& dir, V3D & base1, V3D & base2)
{
    // find base vector in the local tangent space
    base1 = dir.cross(V3D(1.0, 0, 0));
    if (dir(0) == 1.0)
        base1 = dir.cross(V3D(0, 0, 1.0));
    base1.normalize();
    base2 = dir.cross(base1);
    base2.normalize();
}

double threshold = 0.001;
void refineNormalAndCenterDWithCov(const V3D& normal_input, const std::vector<pointWithCov> &points,
                                   const V3D & center_input, V3D & normal_out, V3D & center_out)
{
    bool is_converge = true;
    V3D normal = normal_input;
    V3D center = center_input;
    int i = -1;
    for(; i < refine_maximum_iter; ++i)
    {
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
            A.row(1) = tangent_var_inv * bn1.transpose();
            A.row(2) = tangent_var_inv * bn2.transpose();
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
        normal += dn;
        normal.normalize();

        center += normal * d_w(2);

        for (int j = 0; j < 3; ++j) {
            if (d_w(j) > threshold) {
                is_converge = false;
                break;
            }
        }

        if (is_converge)
            break;
    }
//        ROS_INFO("%d iter normal old [%f %f %f] new [%f %f %f]", i, normal_input(0), normal_input(1), normal_input(2),
//                 normal(0), normal(1), normal(2));
//        ROS_INFO("center old [%f %f %f] new [%f %f %f]", plane->center(0), plane->center(1), plane->center(2),
//                 center(0), center(1), center(2));
    normal_out = normal;
    center_out = center;
}
#endif //SIM_PLANE_SIM_PLANE_H
