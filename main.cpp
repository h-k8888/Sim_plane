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
bool refine_normal_en, incre_cov_en, incre_derivative_en;
//double incre_points = 20;


//plane parameters
V3D normal;
double d;
V3D b1, b2;

//string dir("/home/autolab/Sim_plane/");
string dir("/home/hk/CLionProjects/Sim_plane/");

string cfg_file = dir + "cfg.ini";
string lambda_cov_file  = dir + "lambda_cov";
string nq_cov_file = dir + "n_q_cov";
string time_cost_file = dir + "time_cost";

vector<V4D> cloud, lidars;
vector<V3D> cloud_3D;
vector<M3D> points_cov;
vector<vector<V4D>> cloud_per_lidar;
vector<double> mean_incidents;
vector<V3D> normal_per_lidar;
vector<vector<pointWithCov>> pointWithCov_per_lidar;
vector<pointWithCov> pointWithCov_all;

// incremental cov
double lambda_cov_threshold, normal_cov_threshold;
int num_points_incre_min, num_points_incre_interval;
bool print_lambda_cov_diff, print_nq_cov_diff;

//output 
vector<int> num_points_output;
vector<vector<double>> lambda_cov_gt, lambda_cov_incre_output;
vector<M6D> nq_cov_gt, nq_cov_incre_output;
vector<double> lambda_cov_time_std, lambda_cov_time_incre;
vector<double> nq_cov_time_std, nq_cov_time_incre;

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

void generateCloudOnPlane(const V3D & lidar, const V3D & normal_input, vector<V3D>& cloud,
                          vector<M3D> & points_cov, double & incident_out)
{
    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> gaussian_noise(0.0, noise_stddev); //plane

    V3D b1_tmp, b2_tmp;
    findLocalTangentBases(normal_input, b1_tmp, b2_tmp);

    double incident = 0.0;
    cloud.resize(num_points_per_lidar);
    points_cov.resize(num_points_per_lidar);
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

        points_cov[i] = calcBodyCov(ray, p2lidar, range_stddev, bearing_stddev_deg);
    }
    incident_out = incident;
}

void generateCloudOnPlaneWithRangeBearing(const V3D & lidar, const V3D & normal_input, vector<V3D>& cloud,
                                          vector<M3D> & points_cov, double & incident_out)
{
    std::normal_distribution<double> gaussian_plane(0.0, plane_width); //plane
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> bearing_noise(0.0, bearing_stddev); //bearing

    V3D b1_tmp, b2_tmp;
    findLocalTangentBases(normal_input, b1_tmp, b2_tmp);

    cloud.resize(num_points_per_lidar);
    points_cov.resize(num_points_per_lidar);
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

        points_cov[i] = calcBodyCov(ray, p2lidar, range_stddev, bearing_stddev_deg);

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
                   const vector<vector<M3D>> points_cov_tmp, vector<V4D>& cloud, vector<V4D>& lidars,
                   vector<M3D>& points_cov, vector<vector<V4D>> & cloud_per_lidar)
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

        const vector<M3D> & cov_i = points_cov_tmp[lidar_id];
        points_cov.insert(points_cov.end(), cov_i.begin(), cov_i.end());

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
    vector<vector<M3D>> points_cov_tmp(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        generateCloudOnPlane(lidars_tmp[i], normal_per_lidar[i], cloud_per_lidar_tmp[i],
                             points_cov_tmp[i], mean_incidents[i]);
    }

    recordLidarID(lidars_tmp, cloud_per_lidar_tmp, points_cov_tmp,
                  cloud, lidars, points_cov, cloud_per_lidar);
}

void cloud2lidarWithRangeAndBearing()
{
    vector<V3D> lidars_tmp;
    generateLidar(lidars_tmp);

//    vector<V3D> normals_tmp;
    generatePerturbedNormal(normal_per_lidar);

    vector<vector<V3D>> cloud_per_lidar_tmp(num_lidar);
    mean_incidents.resize(num_lidar);
    vector<vector<M3D>> points_cov_tmp(num_lidar); // todo
    for (int i = 0; i < num_lidar; ++i) {
        generateCloudOnPlaneWithRangeBearing(lidars_tmp[i], normal_per_lidar[i], cloud_per_lidar_tmp[i],
                                             points_cov_tmp[i], mean_incidents[i]);
    }

    recordLidarID(lidars_tmp, cloud_per_lidar_tmp, points_cov_tmp,
                  cloud, lidars, points_cov, cloud_per_lidar);
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
    refine_normal_en = tag_settting.get<bool>("refine_normal_en", false);
//    incre_points = tag_settting.get<double>("incre_points", 0);
    incre_cov_en = tag_settting.get<bool>("incre_cov_en", false);

    noise_type = tag_settting.get<int>("noise_type", 1);
    printf("plane params:\n");
    printf("noise_type: %d\nnormal_pert: %.3f\nrange_stddev: %.3f\nbearing_stddev_deg: %.3f\n",
           noise_type, normal_pert, range_stddev, bearing_stddev_deg);
    printf("incident_max: %.3f\nincident_cov_max: %.3f\nincident_cov_scale:%.3f\nrefine_maximum_iter: %d\n",
           incident_max, incident_cov_max, incident_cov_scale, refine_maximum_iter);

    tag_settting = m_pt.get_child("incremental_derivative");
    incre_derivative_en = tag_settting.get<bool>("incre_derivative_en", false);
    num_points_incre_min = tag_settting.get<int>("num_points_incre_min", 100);
    num_points_incre_interval = tag_settting.get<int>("num_points_incre_interval", 100);
    lambda_cov_threshold = tag_settting.get<double>("lambda_cov_threshold", 0.0);
    normal_cov_threshold = tag_settting.get<double>("normal_cov_threshold", 0.0);
    print_lambda_cov_diff = tag_settting.get<bool>("print_lambda_cov_diff", true);
    print_nq_cov_diff = tag_settting.get<bool>("print_nq_cov_diff", true);

    printf("\nincremental params:\n");
    printf("num_points_incre_min: %d\nnum_points_incre_interval: %d\nlambda_cov_threshold: %e\nnormal_cov_threshold: %e\n\n",
           num_points_incre_min, num_points_incre_interval, lambda_cov_threshold, normal_cov_threshold);

    lidar_width = plane_width * 3.0;
    bearing_stddev = DEG2RAD(bearing_stddev_deg);
    incident_cos_min = cos(incident_max / 180.0 * M_PI);

    normal_per_lidar.resize(num_lidar);
}

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

void saveLambdaCovFile()
{
    assert(num_points_output.size() == lambda_cov_gt.size());
    assert(num_points_output.size() == lambda_cov_incre_output.size());

    printf("\n..............Saving Lambda Cov................\n");
    printf("Lambda cov file: %s\n", lambda_cov_file.c_str());
    ofstream of(lambda_cov_file);
    if (of.is_open())
    {
        of << "points lambda_1_BALM lambda_2_BALM lambda_3_BALM lambda_1_LUFA lambda_2_LUFA lambda_3_LUFA" << endl;
        of.setf(ios::scientific, ios::floatfield);
        of.precision(6);
        for (int i = 0; i < (int)num_points_output.size(); ++i) {
            of<< num_points_output[i] << " "
              << lambda_cov_gt[i][0] << " " << lambda_cov_gt[i][1] << " " << lambda_cov_gt[i][2] << " "
              << lambda_cov_incre_output[i][0] << " "
              << lambda_cov_incre_output[i][1] << " "
              << lambda_cov_incre_output[i][2] << endl;
        }
        of.close();
    }
}

void saveNormalCenterCovFile()
{
    assert(num_points_output.size() == nq_cov_gt.size());
    assert(num_points_output.size() == nq_cov_incre_output.size());

    printf("\n..............Saving normal center Cov................\n");
    printf("normal center cov file: %s\n", nq_cov_file.c_str());
    ofstream of(nq_cov_file);
    if (of.is_open())
    {
        of << "points normal_cov_trace_BALM normal_cov_trace_LUFA "
            << "center_cov_trace_BALM center_cov_trace_LUFA" <<endl;
        of.setf(ios::scientific, ios::floatfield);
        of.precision(6);
        for (int i = 0; i < (int)num_points_output.size(); ++i) {
            M3D  n_cov_gt =  nq_cov_gt[i].block<3, 3>(0, 0);
            M3D  q_cov_gt =  nq_cov_gt[i].block<3, 3>(3, 3);
            M3D  n_cov_incre =  nq_cov_incre_output[i].block<3, 3>(0, 0);
            M3D  q_cov_incre =  nq_cov_incre_output[i].block<3, 3>(3, 3);
            of<< num_points_output[i] << " "
              << n_cov_gt.trace() << " " << n_cov_incre.trace() << " "
              << q_cov_gt.trace() << " " << q_cov_incre.trace() << endl;
        }
        of.close();
    }
}

void saveTimeCostFile()
{
    assert(num_points_output.size() == lambda_cov_time_std.size() + 1);
    assert(num_points_output.size() == lambda_cov_time_incre.size() + 1);
    assert(num_points_output.size() == nq_cov_time_std.size() + 1);
    assert(num_points_output.size() == nq_cov_time_incre.size() + 1);

    printf("\n..............Saving Time Cost................\n");
    printf("time cost file: %s\n", time_cost_file.c_str());
    ofstream of(time_cost_file);
    if (of.is_open())
    {
        of << "points lambda_BALM lambda_LUFA normal_center_BALM normal_center_LUFA\n";
        of.setf(ios::scientific, ios::floatfield);
        of.precision(6);

        of<< num_points_output[0] << " "
          << lambda_cov_time_std[0] << " " << lambda_cov_time_incre[0] << " "
          << nq_cov_time_std[0] << " " << nq_cov_time_incre[0] << endl;

        for (int i = 1; i < (int)num_points_output.size(); ++i) {
            int t = i - 1;
            of<< num_points_output[i] << " "
              << lambda_cov_time_std[t] << " " << lambda_cov_time_incre[t] << " "
              << nq_cov_time_std[t] << " " << nq_cov_time_incre[t] << endl;
        }
        of.close();
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
            printf("***noise: range, bearing, incident and roughness.***\n");
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

    // do PCASelfAdjoint per lidar
    vector<M3D> eigen_vectors(num_lidar);
    vector<V3D> eigen_values(num_lidar), centroids(num_lidar);
    for (int i = 0; i < num_lidar; ++i) {
        printf("lidar #%d\n", i);
        PCASelfAdjoint(cloud_per_lidar[i], eigen_vectors[i], eigen_values[i], centroids[i]);
        double resudial = point2planeResidual(cloud_per_lidar[i], centroids[i], eigen_vectors[i].col(0));
        double theta = diff_normal(normal, eigen_vectors[i].col(0));
        printf("mean_incident: %f\nnormal diff= %f deg\nsum residual^2: %f\n\n",
               mean_incidents[i], theta / M_PI * 180.0, resudial);
    }

    cloud_3D.resize(cloud.size());
    for (int i = 0; i < cloud.size(); ++i) {
        cloud_3D[i] = cloud[i].head(3);
    }

    M3D eigen_vectors_merged;
    V3D eigen_values_merged, centroid_merged;
    PCASelfAdjoint(cloud, eigen_vectors_merged, eigen_values_merged, centroid_merged);
    double resudial_merged = point2planeResidual(cloud, centroid_merged, eigen_vectors_merged.col(0));
    double theta_merged = diff_normal(normal,eigen_vectors_merged.col(0));
    printf("\n****** PCA SelfAdjoint **********\ncloud merged\nnormal diff: %f deg\nsum residual^2: %f\n",
           theta_merged / M_PI * 180.0, resudial_merged);
    printM(eigen_vectors_merged, "PCA Self Adjoint Eigen Vectors");
    printV(eigen_values_merged, "PCA Self Adjoint Eifen Values");

    V3D pca_normal_merged = eigen_vectors_merged.col(0);
    // to pointWithCov format
    recordCloudWithCov();

    if (refine_normal_en)
    {
        {
            // refine normal
            printf("\n********* refine n **********\n");
            V3D normal_perted = pca_normal_merged;
            perturbNormal(normal_perted);
            V3D normal_refine_n = refineNormal(normal_perted, pointWithCov_all, centroid_merged);
            double resudial_refine_n = point2planeResidual(cloud, centroid_merged, normal_refine_n);
            double theta_refine_n = diff_normal(normal, normal_refine_n);
            printf("\nrefine n\nnormal diff: %f deg\nsum residual^2: %f\n",
                   theta_refine_n / M_PI * 180.0, resudial_refine_n);
            printf("PCA SelfAdjoint\nnormal diff: %f deg\nsum residual^2: %f\n*******************\n",
                   theta_merged / M_PI * 180.0, resudial_merged);
        }

        {
            /// refine normal and d with Cov
            printf("\n********* refine n d1 (Cov)**********\n");
            V3D normal_refine, centroid_refine;
            refineNormalAndCenterDWithCov(pca_normal_merged, pointWithCov_all, centroid_merged,
                                          normal_refine, centroid_refine);
            double resudial_merged_refine = point2planeResidual(cloud, centroid_refine, normal_refine);
            double theta_merged_refine = diff_normal(normal, normal_refine);
            printf("\nrefine n d1\nnormal diff: %f deg\nsum residual^2: %f\n",
                   theta_merged_refine / M_PI * 180.0,
                   resudial_merged_refine);

            printf("PCA SelfAdjoint\nnormal diff: %f deg\nsum residual^2: %f\n*******************\n",
                   theta_merged / M_PI * 180.0, resudial_merged);
        }

        {
            /// refine normal
            printf("\n********* refine n (Cov)**********\n");
            V3D normal_refine_n = refineNormalWithCov(pca_normal_merged, pointWithCov_all, centroid_merged);
            double resudial_refine_n = point2planeResidual(cloud, centroid_merged, normal_refine_n);
            double theta_refine_n = diff_normal(normal, normal_refine_n);
            printf("\nrefine n\nnormal diff: %f deg\nsum residual^2: %f\n",
                   theta_refine_n / M_PI * 180.0, resudial_refine_n);

            printf("PCA SelfAdjoint\nnormal diff: %f deg\nsum residual^2: %f\n*******************\n",
                   theta_merged / M_PI * 180.0, resudial_merged);
        }
    }
    // refine normal

    M3D cloud_cov;
    V3D cloud_center;
    M3D eigen_vec_tmp;
    V3D eigen_values_tmp;
    M6D normal_center_cov_tmp;

    // test incremental Cov
    if (incre_cov_en || incre_derivative_en)
    {
        printf("\n****** incremental **********\n");
        M3D cov_start;
        V3D center_start;
        vector<V4D> cloud_start(cloud.begin(), cloud.begin() + num_points_incre_min);
        calcCovMatrix(cloud_start, cov_start, center_start);
        if (incre_cov_en) {
            printM(cov_start, "cov_start");
            printV(center_start, "center_start");
        }

        // set variance before iterative
        vector<V3D> points_old(num_points_incre_min);
//        printf("points_cov.size() %d", (int)points_cov.size());
        vector<M3D> points_cov_old(num_points_incre_min);
        for (int i = 0; i < points_old.size(); ++i) {
            points_old[i] = cloud[i].head(3);
            points_cov_old[i] = points_cov[i];
        }
        M3D eigen_vec_old;
        V3D eigen_values_old, center_old;
        PCASelfAdjoint(points_old, eigen_vec_old, eigen_values_old, center_old);
//            vector<V3D> Jpi_old;
        vector<M3D> Jpi_old;
        vector<double> lambda_cov_old;
        JacobianLambda(points_old, eigen_vec_old, center_old, Jpi_old);
        calcLambdaCov(points_cov_old, Jpi_old, lambda_cov_old);

        // indentity for test
        vector<M3D> points_cov_I_old(points_old.size(), M3D::Identity());
        vector<double> lambda_cov_I_old;
        calcLambdaCov(points_cov_I_old, Jpi_old, lambda_cov_I_old);

        M3D cov;
        V3D center;
        calcCovMatrix(cloud, cov, center);
        if (incre_cov_en) {
            printM(cov, "\ncov n");
            printV(center, "center n");
        }

        vector<M63D> Jnq_p_old;
        M6D n_q_cov_old;
        CovEigenSolverNormalCov(points_old, points_cov_old, eigen_vec_old, eigen_values_old, Jnq_p_old, n_q_cov_old);

        // for output
        num_points_output.push_back(points_old.size());
        lambda_cov_gt.push_back(lambda_cov_old);
        lambda_cov_incre_output.push_back(lambda_cov_old);
        nq_cov_gt.push_back(n_q_cov_old);
        nq_cov_incre_output.push_back(n_q_cov_old);

        double m = cloud.size();
        M3D cov_incre = cov_start;
        V3D center_incre = center_start;
        int points_size_nqcov = num_points_incre_min;
        for (int i = num_points_incre_min; i < m; ++i) {
            TicToc t_cc_inc;
            double n = i + 1;
            const V3D &xn = cloud[i].head(3);
            const M3D & point_cov_i = points_cov[i];
            V3D xn_mn_1 = xn - center_incre;
            cov_incre = (n - 1) / n * (cov_incre + (xn_mn_1 * xn_mn_1.transpose()) / n);
            center_incre = center_incre / n * (n - 1) + xn / n;
            double t_incre1 = t_cc_inc.toc();

            if (incre_derivative_en)
            {
                printf("\n***** add point #%d *****\n", i);
                vector<V3D> points_new = points_old;
                points_new.push_back(xn);
                vector<M3D> points_cov_new = points_cov_old;
                points_cov_new.push_back(point_cov_i);
                vector<M3D> points_cov_I(points_new.size(), M3D::Identity()); // for test
                int num_points_now_last_std = points_new.size() - points_size_nqcov;

                /// PCA SelfAdjoint for all new points
                M3D eigen_vec_new;
                V3D eigen_values_new, center_new;
//                PCASelfAdjoint(points_new, eigen_vec_new, eigen_values_new, center_new);
                PCASelfAdjoint(cov_incre, eigen_vec_new, eigen_values_new);
                center_new = center_incre;

                /// standardized formula for calculating lambda covariance
//                vector<V3D> Jpi_new;
                vector<M3D> Jpi_new;
                M3D cov_new;
                vector<double> lambda_cov_std;
                TicToc t_std;
                calcCloudCov(points_new, cov_new, center_new); // calc cov and center first
                double t_std1 = t_std.toc();
                JacobianLambda(points_new, eigen_vec_new, center_new, Jpi_new);
                double t_std2 = t_std.toc();
                // compute lambda cov
                calcLambdaCov(points_cov_new, Jpi_new, lambda_cov_std);
                double t_l_std3 = t_std.toc();
                printf("std lambda cov cost: %fms\ncenter & cov: %f derivatie %f lambda cov: %f\n",
                       t_l_std3, t_std1, t_std2 - t_std1, t_l_std3 - t_std2);
                // cov as I for test only
                vector<double> lambda_cov_I_new;
                calcLambdaCov(points_cov_I, Jpi_new, lambda_cov_I_new); // in fact, JtJ

                vector<M3D> Jpi_incre;
                vector<double> lambda_cov_incre;
                vector<double> lambda_cov_I_incre;

                bool calc_std_form = false;
                if (num_points_now_last_std >= num_points_incre_interval)
                {
                    JacobianLambda(points_new, eigen_vec_new, center_new, Jpi_new);
                    calcLambdaCov(points_cov_new, Jpi_new, lambda_cov_incre);
                    points_size_nqcov = i + 1;
                    calc_std_form = true;
                    printf("******reach max incremental interval.******\n");
                    printf("******compute lambda cov in std form******\n");
                    lambda_cov_time_incre.push_back(t_incre1 + t_l_std3 - t_std1);
                }
                else {
                    TicToc t_l_incre;
                    double d_lambda0 = incrementalJacobianLambda(points_new, eigen_vec_old, center_old,
                                                                 eigen_vec_new, center_new, Jpi_incre);
                    double t_de_incre1 = t_l_incre.toc();
                    if (d_lambda0 <= lambda_cov_threshold) {
                        calcLambdaCovIncremental(points_cov_new, Jpi_incre, lambda_cov_old, lambda_cov_incre);
                        double t_de_incre2 = t_l_incre.toc();
                        printf("incremental cost: %fms\ncenter & cov %f derivatie %f lambda cov: %f\n",
                               t_de_incre2 + t_incre1, t_incre1, t_de_incre1, t_de_incre2 - t_de_incre1);
                        lambda_cov_time_incre.push_back(t_de_incre2 + t_incre1);

//                    calcLambdaCovIncremental(points_cov_I, Jpi_incre, lambda_cov_I_old, lambda_cov_I_incre);
                    } else {
                        calcLambdaCov(points_cov_new, Jpi_incre, lambda_cov_incre);
                        double t_de_incre2 = t_l_incre.toc();
                        printf("***unvalid*** incremental cost: %fms\ncenter & cov %f derivatie %f lambda cov: %f\n",
                               t_de_incre2 + t_incre1, t_incre1, t_de_incre1, t_de_incre2 - t_de_incre1);
                        points_size_nqcov = i + 1;
                        calc_std_form = true;
                        lambda_cov_time_incre.push_back(t_incre1 + t_l_std3 - t_std1);
//                    calcLambdaCov(points_cov_I, Jpi_incre, lambda_cov_I_incre);
                    }
                    printf("d lambda d p_I magnitude: %e\n", d_lambda0);
                }
                double lambda_cov_diff = 0;
                for (int j = 0; j <3; ++j)
                    lambda_cov_diff += lambda_cov_incre[j] - lambda_cov_std[j];
                printf("lambda cov diff sum: %e\n", lambda_cov_diff);
                if (print_lambda_cov_diff) {
                    printf("diff lambda cov(point): incre - standard\n");
                    for (int k = 0; k < 3; ++k) {
                        printf("lambda %d: %e - %e = %e\n", k,
                               lambda_cov_incre[k], lambda_cov_std[k], lambda_cov_incre[k] - lambda_cov_std[k]);
                    }
                }

                // normal cov
                M3D eigen_vec_std;
                V3D eigen_values_std, center_std;
                M6D normal_center_cov_std;
                vector<M63D> Jnq_p_std;
                TicToc t_es;
                CovEigenSolverNormalCov(points_new, points_cov_new, eigen_vec_std, eigen_values_std, Jnq_p_std,
                                        normal_center_cov_std);
                double t_voxelmap_nq = t_es.toc();
//                EigenSolverNormalCov(cov_new, points_new, center_new, eigen_vec_std, eigen_values_std, normal_center_cov_std);
                printf("\nVoxelMap Eigen Solver & normal Cov cost: %fms\n", t_voxelmap_nq);
                nq_cov_time_std.push_back(t_voxelmap_nq);

                M6D normal_center_cov_tmp2;
                M6D n_q_cov_incre;
                vector<M63D> Jnq_p_incre;
                TicToc t_c_es;
//                calcCloudCov(cloud_3D, cloud_cov, cloud_center); // calc cov and center first
                PCAEigenSolver(cov_incre, eigen_vec_tmp, eigen_values_tmp);
                double t_es_ = t_c_es.toc();
//                calcNormalCov(points_new, eigen_vec_tmp, eigen_values_tmp, center_incre, normal_center_cov_tmp2);
//                calcNormalCovIncremental(points_new, eigen_vec_old, eigen_values_old, center_old, Jnq_p_old,
//                                         n_q_cov_old, eigen_vec_new, eigen_values_new, n_q_cov_incre, Jnq_p_incre);

                if (calc_std_form)
                {
                    TicToc t_nc;
                    calcNormalCov(points_new, points_cov_new, eigen_vec_new, eigen_values_new, center_incre, n_q_cov_incre);
                    double time_nc = t_nc.toc();
                    nq_cov_time_incre.push_back(t_incre1 + t_es_ + time_nc);
                    printf("******reach max incremental interval.******\n");
                    printf("******compute normal, center cov in std form******\n");
                    printf("ES: %fms Normal Cov: %fms\n", t_es_, + time_nc);
                }
                else
                {
                    double normal_cov_diff;
                    normal_cov_diff =
                            calcNormalCovIncremental(points_new, points_cov_new, eigen_vec_old, eigen_values_old,
                                                     center_old, n_q_cov_old, eigen_vec_new, eigen_values_new,
                                                     n_q_cov_incre);
                    if (normal_cov_diff > normal_cov_threshold)
                        calcNormalCov(points_new, points_cov_new, eigen_vec_new, eigen_values_new, center_incre,
                                      n_q_cov_incre);
                    double t_c_es_incre = t_c_es.toc();

                    if (normal_cov_diff > normal_cov_threshold)
                        printf("**unvalid*** incremental normal cov cost: %fms\n", t_incre1 + t_c_es_incre);
                    else
                        printf("valid incremental normal cov cost: %fms\n", t_incre1 + t_c_es_incre);
                    nq_cov_time_incre.push_back(t_incre1 + t_c_es_incre);
//                    printf("normal cov diff magnitude: %e\n", normal_cov_diff);
                    printf("Cov: %fms ES: %fms Normal Cov: %fms\n", t_incre1, t_es_, t_c_es_incre - t_es_);

//                for (int j = 0; j < points_new.size(); ++j) {
//                    M3D j_std = Jnq_p_std[j].block<3,3>(0,0);
//                    printM(j_std, to_string(j) + "j_std");
//                    M3D j_incre = Jnq_p_incre[j].block<3,3>(0,0);
//                    printM(j_incre, to_string(j) + "j_incre");
//                    M3D dndp = j_incre - j_std;
//                    printM(dndp, to_string(j) + " dndp diff");
//                }

                    if (print_nq_cov_diff) {
                        printM(normal_center_cov_std, "normal_center_cov_std");
                        printM(n_q_cov_incre, "n_q_cov_incre");
                        M6D n_q_cov_diff = n_q_cov_incre - normal_center_cov_std;
                        printM(n_q_cov_diff, "n_q_cov_diff");
                        printf("normal center cov trace diff: %e\n", (n_q_cov_incre - normal_center_cov_std).trace());
                    }
                }

                // for output
                num_points_output.push_back(points_new.size());
                lambda_cov_gt.push_back(lambda_cov_std);
                lambda_cov_incre_output.push_back(lambda_cov_incre);
                nq_cov_gt.push_back(normal_center_cov_std);
                nq_cov_incre_output.push_back(n_q_cov_incre);
                lambda_cov_time_std.push_back(t_l_std3);

                points_old = points_new;
                points_cov_old = points_cov_new;
                eigen_vec_old = eigen_vec_new;
                eigen_values_old = eigen_values_new;
                center_old = center_new;
//                Jpi_old = Jpi_incre;
                lambda_cov_old = lambda_cov_incre;
                lambda_cov_I_old = lambda_cov_I_incre;
                Jnq_p_old = Jnq_p_incre;
                n_q_cov_old = n_q_cov_incre;
            }
        }

        if (incre_cov_en) {
            printM(cov_incre, "\ncov n incre");
            printV(center_incre, "center_incre");

            M3D cov_diff = (cov - cov_incre);
            printM(cov_diff, "\ncov diff");
            printV((center - center_incre), "center diff");
        }
    }

    saveLambdaCovFile();
    saveNormalCenterCovFile();
    saveTimeCostFile();
}
