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

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
//#include <boost/filesystem.hpp>
#include <unistd.h>


#include "sim_plane.hpp"

// noise experiment
bool noise_en = true;
float noise_mean = 0.0;
float noise_stddev = 0.04;//plane noise along normal
double plane_width = 20.0;
double lidar_width = plane_width * 3.0;

// normal pertubation (rad noise for every lidar pose)
double normal_pert = 0.00; //std
double range_stddev = 0.00;
double bearing_stddev_deg = 0.0;
double bearing_stddev = DEG2RAD(bearing_stddev_deg);

int num_lidar = 10;
int num_points_per_lidar = 40;

//plane parameters
V3D normal;
double d;
V3D b1, b2;

string cfg_file("./cfg.ini");
//boost::filesystem::path cfg("./cfg.ini");
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

void perturbNormal(V3D & normal_input)
{
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> gaussian_normal(0.0, normal_pert); //normal

    normal_input(0) += gaussian_normal(generator);
    normal_input(1) += gaussian_normal(generator);
    normal_input(2) += gaussian_normal(generator);
    normal_input.normalize();
}

void generatePerturbedNormal(vector<V3D> & normals)
{
    normals.resize(num_lidar);
    for (V3D & n : normals) {
        n = normal;
        perturbNormal(n);
    }
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

void generateCloudOnPlane(const V3D & normal_input, vector<V3D>& cloud)
{
    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> gaussian_noise(0.0, noise_stddev); //plane

    V3D b1_tmp, b2_tmp;
    findLocalTangentBases(normal_input, b1_tmp, b2_tmp);

    cloud.resize(num_points_per_lidar);
    for (int i = 0; i < num_points_per_lidar; ++i) {
        double xyz1 = gaussian_plane(generator);
        double xyz2 = gaussian_plane(generator);
        cloud[i] << b1_tmp * xyz1 + b2_tmp * xyz2 + normal_input * gaussian_noise(generator);
    }
}

void generateCloudOnPlaneWithRangeBearing(const V3D & lidar, const V3D & normal_input, vector<V3D>& cloud)
{
    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> range_noise(0.0, range_stddev); //range
    std::normal_distribution<double> bearing_noise(0.0, bearing_stddev); //bearing

    V3D b1_tmp, b2_tmp;
    findLocalTangentBases(normal_input, b1_tmp, b2_tmp);

    cloud.resize(num_points_per_lidar);
    for (int i = 0; i < num_points_per_lidar; ++i) {
        double xyz1 = gaussian_plane(generator);
        double xyz2 = gaussian_plane(generator);
        V3D pi = b1_tmp * xyz1 + b2_tmp * xyz2; // point exactly on the plane

        V3D ray = pi - lidar;
        double p2lidar = ray.norm();
        ray.normalize();
        V3D br1, br2; // local tangent space bases
        findLocalTangentBases(ray, br1, br2);
        // range, bearing
        double range_offset = range_noise(generator);
        double br1_offset = bearing_noise(generator);
        double br2_offset = bearing_noise(generator);

        pi += range_offset * ray + br1 * br1_offset + br2 * br2_offset;
        cloud[i] = pi;
//        cloud[i] << b1_tmp * xyz1 + b2_tmp * xyz2 + normal_input * gaussian_noise(generator);
    }
}

void generateLidar(vector<V3D>& lidars)
{
    lidars.resize(num_lidar);
    std::normal_distribution<double> gaussian_lidar(0.0, lidar_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    for (int i = 0; i < num_lidar; ++i) {
        double xyz1 = gaussian_lidar(generator);
        double xyz2 = gaussian_lidar(generator);
        lidars[i] << b1 * xyz1 + b2 * xyz2 + normal * abs(gaussian_lidar(generator));
//        printV(cloud[i], "point");
    }
}

void recordLidarID(const vector<V3D> & lidars_tmp, const vector<vector<V3D>> cloud_per_lidar_tmp,
                   vector<V4D>& cloud, vector<V4D>& lidars, vector<vector<V4D>> & cloud_per_lidar)
{
    cloud.clear();
    cloud_per_lidar.resize(num_lidar);
    lidars.resize(num_lidar);
    for (int lidar_id = 0; lidar_id < num_lidar; ++lidar_id) {
        lidars[lidar_id].head(3) = lidars_tmp[lidar_id];
        lidars[lidar_id](3) = lidar_id;

        vector<V4D> & cloud_i = cloud_per_lidar[lidar_id];
        cloud_i.resize(num_points_per_lidar);
        for (int j = 0; j < num_points_per_lidar; ++j) {
            cloud_i[j].head(3) = cloud_per_lidar_tmp[lidar_id][j];
            cloud_i[j](3) = lidar_id;
        }
        cloud.insert(cloud.end(), cloud_i.begin(), cloud_i.end());
    }
}

void cloud2lidar(vector<V4D>& cloud, vector<V4D>& lidars, vector<vector<V4D>> & cloud_per_lidar)
{
    vector<V3D> lidars_tmp;
    generateLidar(lidars_tmp);

    vector<V3D> normals_tmp;
    generatePerturbedNormal(normals_tmp);

    vector<vector<V3D>> cloud_per_lidar_tmp(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        generateCloudOnPlane(normals_tmp[i], cloud_per_lidar_tmp[i]);
    }

    recordLidarID(lidars_tmp, cloud_per_lidar_tmp, cloud, lidars, cloud_per_lidar);
}

void cloud2lidarWithRangeAndBearing(vector<V4D>& cloud, vector<V4D>& lidars, vector<vector<V4D>> & cloud_per_lidar)
{
    vector<V3D> lidars_tmp;
    generateLidar(lidars_tmp);

    vector<V3D> normals_tmp;
    generatePerturbedNormal(normals_tmp);

    vector<vector<V3D>> cloud_per_lidar_tmp(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        generateCloudOnPlaneWithRangeBearing(lidars_tmp[i], normals_tmp[i], cloud_per_lidar_tmp[i]);
    }

    recordLidarID(lidars_tmp, cloud_per_lidar_tmp, cloud, lidars, cloud_per_lidar);
}

void readParams()
{
    char buff[250];
    getcwd(buff, 250);
    string flie(buff);
    cout << flie << endl;
    flie  = flie + "../cfg.ini";

    boost::property_tree::ptree m_pt, tag_settting;
    try {
        boost::property_tree::read_ini(flie, m_pt);
    }
    catch (exception e) {
        cout << "open cfg file failed." << endl;
    }

//    tag_settting = m_pt.get_child("sim_plane");
//    noise_mean = tag_settting.get<double>("noise_mean", 0.0);
//    cout << "noise_mean " << noise_mean <<  endl;
//
//    tag_settting = m_pt.get_child("sim_plane");
//    noise_mean = tag_settting.get<double>("noise_mean", 0.0);
//    noise_stddev = tag_settting.get<double>("noise_stddev", 0.0);
//    plane_width = tag_settting.get<double>("plane_width", 0.0);
//
//    normal_pert = tag_settting.get<double>("normal_pert", 0.0);
//    range_stddev = tag_settting.get<double>("range_stddev", 0.0);
//    bearing_stddev_deg = tag_settting.get<double>("bearing_stddev_deg", 0.0);
//
//    num_lidar = tag_settting.get<int>("num_lidar", 0);
//    num_points_per_lidar = tag_settting.get<int>("num_points_per_lidar", 0.0);
//
//
//    lidar_width = plane_width * 3.0;
//    bearing_stddev = DEG2RAD(bearing_stddev_deg);
}

int main(int argc, char** argv) {
    cout << "hello world" << endl;
//    cout << cfg << endl;


    readParams();

    generatePlane();
    printV(normal, "normal");
    cout << "d = " << d << endl;
    findLocalTangentBases(normal, b1, b2);

//    vector<V3D> cloud;
//    generateCloudOnPlane(cloud, 500);
//
//    vector<V3D> lidars;
//    generateLidar(lidars, 10);

    vector<V4D> cloud, lidars;
    vector<vector<V4D>> cloud_per_lidar;
//    cloud2lidar(cloud, lidars, cloud_per_lidar);
    cloud2lidarWithRangeAndBearing(cloud, lidars, cloud_per_lidar);

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
    vector<V3D> eigen_values(num_lidar), controids(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        PCA(cloud_per_lidar[i], eigen_vectors[i], eigen_values[i], controids[i]);
        double resudial = point2planeResidual(cloud_per_lidar[i],  controids[i], eigen_vectors[i].col(0));
        double theta = diff_normal(normal, eigen_vectors[i].col(0));
        printf("lidar #%d normal diff= %f deg, sum residual^2: %f\n\n", i, theta / M_PI * 180.0, resudial);
    }

    M3D eigen_vectors_merged;
    V3D eigen_values_merged, controid_merged;
    PCA(cloud, eigen_vectors_merged, eigen_values_merged, controid_merged);
    double resudial_merged = point2planeResidual(cloud,  controid_merged, eigen_vectors_merged.col(0));
    double theta_merged = diff_normal(normal,eigen_vectors_merged.col(0));
    printf("cloud merged normal diff: %f deg, sum residual^2: %f\n\n", theta_merged / M_PI * 180.0,
           resudial_merged);
    
    
}