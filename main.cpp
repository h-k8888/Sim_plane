//
// Created by hk on 1/19/24.
//

#include <iostream>
#include <math.h>
#include <thread>
#include <fstream>
#include <random>
#include <fstream>
#include <string>

#include <Eigen/Dense>

#include "sim_plane.hpp"



// noise experiment
bool noise_en = true;
float noise_mean = 0.0;
float noise_stddev = 0.0;//plane noise along normal
double plane_width = 20.0;
double lidar_width = plane_width * 3.0;

// normal pertubation
double normal_per = 0.05;

//plane parameters
V3D normal;
double d;
V3D b1, b2;

void generatePlane()
{
    std::normal_distribution<double> gaussian_noise(0.0, 1.0);
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    normal << gaussian_noise(generator),  gaussian_noise(generator), gaussian_noise(generator);
    normal.normalize();

    std::normal_distribution<double> gaussian_noise_1(0, 50);
    d =  gaussian_noise_1(generator);
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

void generateCloudOnPlane(vector<V3D>& cloud, int n = 50)
{
    cloud.resize(n);
    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> gaussian_noise(0.0, noise_stddev); //plane
    for (int i = 0; i < n; ++i) {
        double xyz1 = gaussian_plane(generator);
        double xyz2 = gaussian_plane(generator);
        cloud[i] << b1 * xyz1 + b2 * xyz2 + normal * gaussian_noise(generator);
//        printV(cloud[i], "point");
    }
}

void generateLidar(vector<V3D>& lidars, int n = 30)
{
    lidars.resize(n);
    std::normal_distribution<double> gaussian_lidar(0.0, lidar_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

//    std::normal_distribution<double> gaussian_noise(0.0, noise_stddev); //plane
    for (int i = 0; i < n; ++i) {
        double xyz1 = gaussian_lidar(generator);
        double xyz2 = gaussian_lidar(generator);
        lidars[i] << b1 * xyz1 + b2 * xyz2 + normal * abs(gaussian_lidar(generator));
//        printV(cloud[i], "point");
    }
}

void cloud2lidar(vector<V4D>& cloud, vector<V4D>& lidars, vector<vector<V4D>> & cloud_per_lidar, int points_per_lidar, int num_lidar)
{
    std::normal_distribution<double> gaussian_lidar(0.0, lidar_width); //lidar
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> gaussian_normal(0.0, normal_per); //normal

    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane width
//    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
//    std::default_random_engine generator(seed);

    std::normal_distribution<double> gaussian_noise(0.0, noise_stddev); //plane noise along normal

    lidars.resize(num_lidar);
    cloud.resize(num_lidar * points_per_lidar);
    cloud_per_lidar.resize(num_lidar);
    int point_size = cloud.size() / lidars.size();
    int point_id = 0;
    for (int i = 0; i < num_lidar; ++i) {
        // plane pertubation
        double xyz1 = gaussian_lidar(generator);
        double xyz2 = gaussian_lidar(generator);
        lidars[i].head(3) = b1 * xyz1 + b2 * xyz2 + normal * abs(gaussian_lidar(generator));
        lidars[i](3) = i;

        V3D normal_tmp = normal;
        normal_tmp(0) += gaussian_normal(generator);
        normal_tmp(1) += gaussian_normal(generator);
        normal_tmp(2) += gaussian_normal(generator);
        normal_tmp.normalize();

        double d_tmp = d;
        d_tmp += gaussian_normal(generator);

        V3D b1_tmp, b2_tmp;
        findLocalTangentBases(normal_tmp, b1_tmp, b2_tmp);

//        cloud_per_lidar[i].push_back(lidars[i]);

        for (int j = 0; j < points_per_lidar; ++j) {
            double xyz1 = gaussian_plane(generator);
            double xyz2 = gaussian_plane(generator);
            cloud[point_id].head(3) = b1_tmp * xyz1 + b2_tmp * xyz2 + normal_tmp * gaussian_noise(generator);
//            cloud[point_id].head(3) = b1 * xyz1 + b2 * xyz2;
            cloud[point_id](3) = i;
            cloud_per_lidar[i].push_back(cloud[point_id]);
            point_id++;
        }
    }
}

int main(int argc, char** argv) {
    cout << "hello world" << endl;

    generatePlane();
    printV(normal, "normal");
    cout << "d = " << d << endl;

    findLocalTangentBases(normal, b1, b2);

//    vector<V3D> cloud;
//    generateCloudOnPlane(cloud, 500);
//
//    vector<V3D> lidars;
//    generateLidar(lidars, 10);

    int num_points_per_lidar = 40;
    int num_lidar = 10;
    vector<V4D> cloud, lidars;
    vector<vector<V4D>> cloud_per_lidar;
    cloud2lidar(cloud, lidars, cloud_per_lidar, num_points_per_lidar, num_lidar);

    string file("/tmp/cloud.txt");
    saveCloud(cloud, file);
    string file_lidar("/tmp/lidar.txt");
    saveCloud(lidars, file_lidar);

    vector<V4D> cloud_lidar_combine(cloud);
    cloud_lidar_combine.insert(cloud_lidar_combine.end(), lidars.begin(), lidars.end());
    string combine("/tmp/cloud_lidar_combine.txt");
    saveCloud(cloud_lidar_combine, combine);

    string file_cloud_lidar("/tmp/cloud_lidar");
    saveCloud(cloud_per_lidar, file_cloud_lidar);

    // do PCA per lidar
    vector<M3D> eigen_vectors(num_lidar);
    vector<V3D> eigen_values(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        PCA(cloud_per_lidar[i], eigen_vectors[i], eigen_values[i]);
        double theta = diff_normal(normal, eigen_vectors[i].col(0));
        printf("lidar #%d normal diff: %f deg\n\n", i, theta / M_PI * 180.0);
    }


}