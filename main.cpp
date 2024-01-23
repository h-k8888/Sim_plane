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
int noise_type = 1;
float noise_mean = 0.0;
float noise_stddev = 0.04;//plane noise along normal
double plane_width = 20.0;
double lidar_width = plane_width * 3.0;

int num_lidar = 10;
int num_points_per_lidar = 40;

// normal pertubation (rad noise for every lidar pose)
double normal_pert = 0.00; //std
double range_stddev = 0.00;
double bearing_stddev_deg = 0.0;
double bearing_stddev = DEG2RAD(bearing_stddev_deg);

double incident_max = 75.0; //degree
double incident_cos_min = cos(incident_max / 180.0 * M_PI);
double incident_cov_max, incident_cov_scale;

int refine_maximum_iter;

//plane parameters
V3D normal;
double d;
V3D b1, b2;

string cfg_file("/home/hk/CLionProjects/Sim_plane/cfg.ini");

vector<V4D> cloud, lidars;
vector<vector<V4D>> cloud_per_lidar;
vector<double> mean_incidents;
vector<V3D> normal_per_lidar;
vector<vector<pointWithCov>> pointWithCov_per_lidar;
vector<pointWithCov> pointWithCov_all;

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

void generateCloudOnPlane(const V3D & lidar, const V3D & normal_input, vector<V3D>& cloud, double & incident_out)
{
    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> gaussian_noise(0.0, noise_stddev); //plane

    V3D b1_tmp, b2_tmp;
    findLocalTangentBases(normal_input, b1_tmp, b2_tmp);

    double incident = 0.0;
    cloud.resize(num_points_per_lidar);
    for (int i = 0; i < num_points_per_lidar; ++i) {
        double xyz1 = gaussian_plane(generator);
        double xyz2 = gaussian_plane(generator);
        cloud[i] << b1_tmp * xyz1 + b2_tmp * xyz2 + normal_input * gaussian_noise(generator);

        // incident
        V3D ray = cloud[i] - lidar;
        double p2lidar = ray.norm();
        ray.normalize();
        double cos_incident = abs(normal.dot(ray));
        double incident_tmp = acos(cos_incident) / M_PI * 180.0;
        incident += incident_tmp / num_points_per_lidar;
    }
    incident_out = incident;
}

//// ray: unit vector
//double calcIncidentCovScale(const V3D & ray, const double & dist, const V3D& normal)
//{
////    static double angle_rad = DEG2RAD(angle_cov);
//    double cos_incident = abs(normal.dot(ray));
//    cos_incident = max(cos_incident, incident_cos_min);
//    double sin_incident = sqrt(1 - cos_incident * cos_incident);
////    return dist * sin_incident / cos_incident * bearing_stddev; // range * tan(incident) * sigma_angle
//    double sigma_a = dist * sin_incident / cos_incident * bearing_stddev; // range * tan(incident) * sigma_angle
//    return min(incident_cov_max, incident_cov_scale * sigma_a * sigma_a); // scale * sigma_a^2
//}

void generateCloudOnPlaneWithRangeBearing(const V3D & lidar, const V3D & normal_input, vector<V3D>& cloud,
                                          double & incident_out)
{
    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> bearing_noise(0.0, bearing_stddev); //bearing

    V3D b1_tmp, b2_tmp;
    findLocalTangentBases(normal_input, b1_tmp, b2_tmp);

    cloud.resize(num_points_per_lidar);
    double incident = 0.0;
    for (int i = 0; i < num_points_per_lidar; ++i) {
        double xyz1 = gaussian_plane(generator);
        double xyz2 = gaussian_plane(generator);
        V3D pi = b1_tmp * xyz1 + b2_tmp * xyz2; // point exactly on the plane

        V3D ray = pi - lidar;
        double p2lidar = ray.norm();
        ray.normalize();
        V3D br1, br2; // local tangent space bases
        findLocalTangentBases(ray, br1, br2);

        // incident
        double cos_incident = abs(normal.dot(ray));
        double incident_tmp = acos(cos_incident) / M_PI * 180.0;
        incident += incident_tmp / num_points_per_lidar;

        double range_cov = range_stddev * range_stddev;
        if (noise_type == 3) {
//            cout << "\nrange_cov " << range_cov << endl;
            double incident_cov = calcIncidentCovScale(ray, p2lidar, normal_input);
            range_cov += incident_cov;
//            cout << "range_cov " << range_cov << endl;
        }
        std::normal_distribution<double> range_noise(0.0, sqrt(range_cov)); //range

        // range, bearing
        double range_offset = range_noise(generator);
        double br1_offset = bearing_noise(generator);
        double br2_offset = bearing_noise(generator);

        pi += range_offset * ray + br1 * br1_offset + br2 * br2_offset;
        cloud[i] = pi;
//        cloud[i] << b1_tmp * xyz1 + b2_tmp * xyz2 + normal_input * gaussian_noise(generator);
    }
    incident_out = incident;
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

void cloud2lidar()
{
    vector<V3D> lidars_tmp;
    generateLidar(lidars_tmp);

//    vector<V3D> normals_tmp;
    generatePerturbedNormal(normal_per_lidar);

    mean_incidents.resize(num_lidar);
    vector<vector<V3D>> cloud_per_lidar_tmp(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        generateCloudOnPlane(lidars_tmp[i], normal_per_lidar[i], cloud_per_lidar_tmp[i], mean_incidents[i]);
    }

    recordLidarID(lidars_tmp, cloud_per_lidar_tmp, cloud, lidars, cloud_per_lidar);
}

void cloud2lidarWithRangeAndBearing()
{
    vector<V3D> lidars_tmp;
    generateLidar(lidars_tmp);

//    vector<V3D> normals_tmp;
    generatePerturbedNormal(normal_per_lidar);

    vector<vector<V3D>> cloud_per_lidar_tmp(num_lidar);
    mean_incidents.resize(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        generateCloudOnPlaneWithRangeBearing(lidars_tmp[i], normal_per_lidar[i], cloud_per_lidar_tmp[i], mean_incidents[i]);
    }

    recordLidarID(lidars_tmp, cloud_per_lidar_tmp, cloud, lidars, cloud_per_lidar);
}

void readParams()
{
    boost::property_tree::ptree m_pt, tag_settting;
    try {
        boost::property_tree::read_ini(cfg_file, m_pt);
    }
    catch (exception e) {
        cout << "open cfg file failed." << endl;
    }
    tag_settting = m_pt.get_child("sim_plane");
    noise_mean = tag_settting.get<double>("noise_mean", 0.0);
    noise_stddev = tag_settting.get<double>("noise_stddev", 0.0);
    plane_width = tag_settting.get<double>("plane_width", 0.0);
    num_lidar = tag_settting.get<int>("num_lidar", 0);
    num_points_per_lidar = tag_settting.get<int>("num_points_per_lidar", 0.0);
    printf("noise_mean: %.3f\nnoise_stddev: %.3f\nplane_width %.3f\n"
           "num_lidar %d\nnum_points_per_lidar %d\n", noise_mean, noise_stddev,
           plane_width, num_lidar, num_points_per_lidar);

    normal_pert = tag_settting.get<double>("normal_pert", 0.0);
    range_stddev = tag_settting.get<double>("range_stddev", 0.0);
    bearing_stddev_deg = tag_settting.get<double>("bearing_stddev_deg", 0.0);
    incident_max = tag_settting.get<double>("incident_max", 0.0);
    incident_cov_max = tag_settting.get<double>("incident_cov_max", 0.0);
    incident_cov_scale = tag_settting.get<double>("incident_cov_scale", 0.0);
    refine_maximum_iter = tag_settting.get<int>("refine_maximum_iter", 0);

    noise_type = tag_settting.get<int>("noise_type", 1);
    printf("noise_type: %d\nnormal_pert: %.3f\nrange_stddev: %.3f\nbearing_stddev_deg: %.3f\n",
           noise_type, normal_pert, range_stddev, bearing_stddev_deg);
    printf("incident_max: %.3f\nincident_cov_max: %.3f\nincident_cov_scale:%.3f\nrefine_maximum_iter: %d\n",
           incident_max, incident_cov_max, incident_cov_scale, refine_maximum_iter);


    lidar_width = plane_width * 3.0;
    bearing_stddev = DEG2RAD(bearing_stddev_deg);
    incident_cos_min = cos(incident_max / 180.0 * M_PI);

    normal_per_lidar.resize(num_lidar);
}

// todo record as type pointWithCov
void recordPWC(const vector<V4D>& cloud_input, const V3D & n, const V4D & lidar, vector<pointWithCov> & cloud_out)
{
    cloud_out.resize(cloud_input.size());
    for (int i = 0; i < (int)cloud_input.size(); ++i) {
        const V4D & p_world = cloud_input[i];
        pointWithCov & pv = cloud_out[i];
        pv.point << p_world(0), p_world(1), p_world(2);
        pv.normal << n(0), n(1), n(2); //TODO
        pv.ray = pv.point - lidar.head(3);
        pv.p2lidar = pv.ray.norm();
        pv.ray.normalize();
//        pv.roughness_cov = calcRoughCovScale(feats_down_body->points[i].intensity); //TODO
        pv.roughness_cov = 0.0;
        pv.tangent_cov = pow(DEG2RAD(range_stddev), 2) * pv.p2lidar * pv.p2lidar;
    }
}

void recordCloudWithCov()
{
    pointWithCov_per_lidar.resize(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        recordPWC(cloud_per_lidar[i], normal_per_lidar[i], lidars[i], pointWithCov_per_lidar[i]);
        pointWithCov_all.insert(pointWithCov_all.end(),
                                pointWithCov_per_lidar[i].begin(), pointWithCov_per_lidar[i].end());
    }
}

int main(int argc, char** argv) {
    cout << "hello world" << endl;

    readParams();

    generatePlane();
    printV(normal, "\nnormal");
    cout << "d: " << d << endl;
    findLocalTangentBases(normal, b1, b2);

//    vector<V3D> cloud;
//    generateCloudOnPlane(cloud, 500);
//
//    vector<V3D> lidars;
//    generateLidar(lidars, 10);

//    vector<V4D> cloud, lidars;
//    vector<vector<V4D>> cloud_per_lidar;
//    vector<double> mean_incidents;
    switch (noise_type) {
        case 1:
        {
            printf("***noise: iostropic.***\n");
            cloud2lidar();
            break;
        }
        case 2:
        {
            printf("***noise: range and bearing.***\n");
            cloud2lidarWithRangeAndBearing();
            break;
        }
        case 3:
        {
            printf("***noise: range, bearing, invident and roughness.***\n");
            cloud2lidarWithRangeAndBearing();
            break;
        }
    }


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
    vector<V3D> eigen_values(num_lidar), centroids(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        PCA(cloud_per_lidar[i], eigen_vectors[i], eigen_values[i], centroids[i]);
        double resudial = point2planeResidual(cloud_per_lidar[i], centroids[i], eigen_vectors[i].col(0));
        double theta = diff_normal(normal, eigen_vectors[i].col(0));
        printf("lidar #%d\nmean_incident: %f\nnormal diff= %f deg\nsum residual^2: %f\n\n",
               i, mean_incidents[i], theta / M_PI * 180.0, resudial);
    }

    M3D eigen_vectors_merged;
    V3D eigen_values_merged, centroid_merged;
    PCA(cloud, eigen_vectors_merged, eigen_values_merged, centroid_merged);
    double resudial_merged = point2planeResidual(cloud, centroid_merged, eigen_vectors_merged.col(0));
    double theta_merged = diff_normal(normal,eigen_vectors_merged.col(0));
    printf("cloud merged\nnormal diff: %f deg\nsum residual^2: %f\n\n", theta_merged / M_PI * 180.0,
           resudial_merged);

    V3D pca_normal_merged = eigen_vectors_merged.col(0);
    // to pointWithCov format
    recordCloudWithCov();
    V3D normal_refine, centroid_refine;
    refineNormalAndCenterDWithCov(pca_normal_merged, pointWithCov_all, centroid_merged,
                                  normal_refine, centroid_refine);
    double resudial_merged_refine = point2planeResidual(cloud, centroid_refine, normal_refine);
    double theta_merged_refine = diff_normal(normal, normal_refine);
    printf("refine\nnormal diff: %f deg\nsum residual^2: %f\n\n", theta_merged_refine / M_PI * 180.0,
           resudial_merged_refine);
}