//
// Created by hk on 1/19/24.
//

#include <iostream>
#include <math.h>
#include <thread>
#include <fstream>
#include <random>
#include <Eigen/Dense>

using namespace std;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;


// noise experiment
bool noise_en = true;
float noise_mean = 0.0;
float noise_stddev = 5.00;

void printV(const V3D & v, const string & s)
{
    cout << s << ":\n" << v.transpose() <<endl;
}

void printM(const M3D & m, const string & s)
{
    cout << s << ":\n" << m <<endl;
}

void generatePlane(V3D & n, double & d)
{
    std::normal_distribution<double> gaussian_noise(noise_mean, noise_stddev);
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    n << gaussian_noise(generator),  gaussian_noise(generator), gaussian_noise(generator);
    n.normalize();

    std::normal_distribution<double> gaussian_noise_1(0, 50);
    d =  gaussian_noise_1(generator);
}

int main(int argc, char** argv) {
    cout << "hello world" << endl;
    V3D normal ;
    double d;
    generatePlane(normal, d);
    printV(normal, "normal");
    cout << "d = " << d << endl;


}