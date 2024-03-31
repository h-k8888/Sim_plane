# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys

import matplotlib.pyplot as plt
from xml.dom.minidom import parse
import numpy as np
import os


def loadData(flieName):
    inFile = open(flieName, 'r')

    distance = []
    incident = []
    reflectivity = []
    intensityOriginal = []
    intensityCorrected = []
    intensityIncidentCorrected = []
    intensityDistanceCorrected = []
    for line in inFile:
        element = line.split(',')
        distance.append(float(element[0]))
        incident.append(float(element[1]))
        reflectivity.append(float(element[2]))
        intensityOriginal.append(float(element[3]))
        intensityCorrected.append(float(element[4]))
        # intensityIncidentCorrected.append(float(element[5]))
        # intensityDistanceCorrected.append(float(element[6]))

    return (distance, incident, reflectivity, intensityOriginal, intensityCorrected
            # , intensityIncidentCorrected, intensityDistanceCorrected
            )


def make_chart(x, y, s):
    plt.plot(x, y, s)
    # plt.scatter(x, y, alpha=0.5)
    plt.title('intensity - theta', fontsize=14)
    plt.xlabel('theta', fontsize=14)
    plt.ylabel('intensity', fontsize=14)
    plt.grid(True)
    # plt.show()


def make_lambda1_chart(lambda_data):
    num_points = lambda_data['points']
    lambda_1_balm = lambda_data['lambda_1_BALM']
    lambda_1_min = min(lambda_1_balm)
    lambda_1_max = max(lambda_1_balm)
    plt.figure(figsize=(9, 3))
    text_size = 14
    plt.title(r'$\lambda_1$ covariance', fontsize=text_size)
    # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), constrained_layout=True)
    # axs[0].set_title('eigenvalues covariance', fontsize=14)
    plt.plot(num_points, lambda_1_balm, label='BALM', alpha=0.6, linewidth=2, color='red')
    plt.plot(num_points, lambda_data['lambda_1_LUFA'], label='LUFA', alpha=0.6, linewidth=2, color='green')
    lambda_inteval = (lambda_1_max - lambda_1_min) / 5.0
    y_ticks = np.arange(lambda_1_min, lambda_1_max + lambda_inteval, lambda_inteval)
    plt.yticks(y_ticks)
    x_ticks = np.arange(min(num_points), max(num_points)+100, 100)
    plt.xticks(x_ticks)

    plt.xlabel('points', fontsize=text_size)
    plt.ylabel('value', fontsize=text_size)
    # l1_label = plt.gca().get_ylabel()
    plt.legend(fontsize=text_size+2)
    plt.grid(True)
    plt.autoscale(enable = True, axis='x', tight=True)
    plt.tight_layout()
    plt.show()


def make_lambda_chart_all(lambda_data):
    num_points = lambda_data['points']
    lambda_1_balm = lambda_data['lambda_1_BALM']
    lambda_1_min = min(lambda_1_balm)
    lambda_1_max = max(lambda_1_balm)
    # plt.title('eigenvalues covariance', fontsize=14)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), constrained_layout=True)
    axs[0].set_title('eigenvalues covariance', fontsize=14)
    axs[0].plot(num_points, lambda_1_balm, color='magenta', label=r'$\lambda_1$_BALM', alpha=0.6)
    axs[0].plot(num_points, lambda_data['lambda_1_LUFA'], color='green', label=r'$\lambda_1$_LUFA', alpha=0.6)
    # axs[0].title('eigenvalues covariance', fontsize=14)
    axs[0].set_xlabel('points', fontsize=10)
    axs[0].set_ylabel(r'$\lambda_1$ value', fontsize=10)
    # axs[0].xlabel('points', fontsize=14)
    # axs[0].ylabel(r'$\lambda_1$ value', fontsize=14)
    # l1_label = plt.gca().get_ylabel()
    axs[0].legend(fontsize=12)
    axs[0].grid(True)
    # plt.show()


    # plt.twinx()

    lambda_2_balm = lambda_data['lambda_2_BALM']
    lambda_2_min = min(lambda_2_balm)
    lambda_2_max = max(lambda_2_balm)
    axs[1].plot(num_points, lambda_2_balm, label=r'$\lambda_2$_BALM')

    lambda_3_balm = lambda_data['lambda_3_BALM']
    lambda_3_min = min(lambda_3_balm)
    lambda_3_max = max(lambda_3_balm)
    axs[1].plot(num_points, lambda_3_balm, label=r'$\lambda_3$_BALM')

    axs[1].plot(num_points, lambda_data['lambda_2_LUFA'], label=r'$\lambda_2$_LUFA')
    axs[1].plot(num_points, lambda_data['lambda_3_LUFA'], label=r'$\lambda_3$_LUFA')

    lambda_min = min(lambda_2_min, lambda_3_min)
    lambda_max = max(lambda_2_max, lambda_3_max)
    l23_label = plt.gca().get_ylabel()
    x_ticks = np.arange(min(num_points), max(num_points)+100, 100)
    # axs[1].xticks(x_ticks)
    lambda_inteval = (lambda_max -lambda_min) / 5.0
    y_ticks = np.arange(lambda_min, lambda_max + lambda_inteval, lambda_inteval)
    # axs[1].yticks(y_ticks)
    axs[1].autoscale(enable = True, axis='y', tight=False)

    # lines, labels = plt.gca().get_legend_handles_labels()
    # first_legend = plt.legend(lines[:2], labels[:2], loc='upper left')
    # plt.gca().add_artist(first_legend)

    # axs[1].title('eigenvalues covariance', fontsize=14)
    # axs[1].xlabel('points', fontsize=14)
    # axs[1].ylabel(r'$\lambda_2$, $\lambda_3$ value', fontsize=14)
    axs[1].set_xlabel('points', fontsize=10)
    axs[1].set_ylabel(r'$\lambda_2$, $\lambda_3$ value', fontsize=10)
    axs[1].grid(True)
    axs[1].legend(fontsize=12)
    plt.show()


def read_txt_data(file_name):
    with open(file_name, 'r') as file:
        header = file.readline().strip().split()
        data_dict = {col: [] for col in header}
        for line in file:
            data = [float(value) for value in line.strip().split()]
            # data = line.strip().split()
            if len(data) == len(header):
                for col, value in zip(header, data):
                    data_dict[col].append(value)
    return data_dict


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(os.getcwd())
    lambda_cov_file = os.getcwd() + "/lambda_cov"
    normal_center_cov_file = os.getcwd() + "/n_q_cov"
    time_cost_file = os.getcwd() + "/time_cost"
    lambda_data = read_txt_data(lambda_cov_file)
    n_q_data = read_txt_data(normal_center_cov_file)
    time_cost_data = read_txt_data(time_cost_file)

    # print(lambda_data)
    make_lambda_chart_all(lambda_data)
    make_lambda1_chart(lambda_data)
    print(lambda_data['points'])

    # print_intensity(theta, intensity, 'g^')
    # print_intensity(theta, intensityCorrected, 'bs')
    # # # print_intensity(distance, intensity, 'g^')
    # # # print_intensity(distance, intensityCorrected, 'ro')
    #
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
