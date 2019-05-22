from django.http import HttpResponse
from django.shortcuts import render
from .BPNetWork import NeuronNetWork
import math
import random
import numpy as np
import matplotlib.pyplot as plt


# 计算神经网络时辅助的函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pd_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def square_error(target, output):
    return 1 / 2 * np.square(target - output)


def pd_error_out(target, output):
    return output - target


def generate_input_and_output_data(input_begin, input_end, function_name):
    output_data_list = []
    input_data_list = list(np.arange(input_begin, input_end, 0.1))  # 训练数据
    input_test_data_list = []  # 测试数据的输入
    output_test_data_list = []  # 测试数据的输出
    for i in range(len(input_data_list)):
        input_test_data_list.append(input_data_list[i] + 0.05)  # 测试数据产生的方式是在原有训练数据的基础上每个值+0.05
    for i in range(len(input_data_list)):
        if function_name is 'sin':
            output_data_list.append(math.sin(input_data_list[i]) + 1)  # y = sin(x) + 1
            output_test_data_list.append(math.sin(input_test_data_list[i]) + 1)  # y = sin(x) + 1
        elif function_name is 'xsquare':
            output_data_list.append(math.pow(input_data_list[i], 2))  # y = x * x
            output_test_data_list.append(math.pow(input_test_data_list[i], 2))  # y = x * x
    return input_data_list, output_data_list, input_test_data_list, output_test_data_list


# 归一化
def generalize_list(input_data_list, output_data_list):
    input_max = -1
    output_max = -1
    for i in range(len(input_data_list)):
        if input_data_list[i] > input_max:
            input_max = input_data_list[i]
        if output_data_list[i] > output_max:
            output_max = output_data_list[i]
    for i in range(len(input_data_list)):
        input_data_list[i] /= input_max
        output_data_list[i] /= output_max
    return input_data_list, output_data_list, input_max, output_max


def de_generalize(data_list, ori_data_list_max):
    """
    反归一化
    :param data_list: 被反归一化的列表
    :param ori_data_list_max: 原列表中的最大值
    :return: 反归一化的列表
    """
    de_data_list = []  # 反归一化后的数据表
    for i in range(len(data_list)):
        de_data_list.append(data_list[i] * ori_data_list_max)
    return de_data_list


# 可视化多项式曲线拟合结果
def draw_fit_curve(origin_xs, target_ys, actual_ys, step_arr, loss_arr):
    plt.cla()
    fig = plt.figure("BP")
    ax1 = fig.add_subplot(121)
    # 分别画出理想的输出和实际的输出
    ax1.plot(origin_xs, target_ys, color='r', label='original data')
    ax1.plot(origin_xs, actual_ys, color='b', label='curved data')
    plt.title(label='BP NetWork')
    ax2 = fig.add_subplot(122)
    ax2.plot(step_arr, loss_arr, color='red', label='error curve')
    plt.title(label='BP error')
    plt.legend()
    plt.pause(0.001)


# Create your views here.
def index(request):
    return render(request, 'index.html')


def train(request):
    # 创建输入数据和输出数据
    # 随机产生输入和输出，是没有归一化的数据
    ori_input_data_list, ori_output_data_list, ori_input_test_data_list, ori_output_test_data_list = generate_input_and_output_data(-5, 5, 'xsquare')

    # 归一化
    # input_max为input_data_list中的最大值,output_max为output_data_list中的最大值
    input_data_list, output_data_list, input_max, output_max = generalize_list(ori_input_data_list,
                                                                          ori_output_data_list)

    global input_test_data_list, output_test_data_list, input_test_max, output_test_max
    input_test_data_list, output_test_data_list, input_test_max, output_test_max = generalize_list(ori_input_test_data_list,
                                                                                                   ori_output_test_data_list)

    # 创建神经网络
    global nn
    nn = NeuronNetWork(input_layer_neuron_num=len(input_data_list), output_layer_neuron_num=len(input_data_list),
                       hidden_layer_neuron_num_list=[3],
                       input_data_list=input_data_list, output_data_list=output_data_list,
                       input_max=input_max, output_max=output_max)

    # 初始化神经网络的w和b
    nn.init_w_b()

    while True:
        nn.iteration += 1

        nn.positive_forward()
        # nn.print_layer_input_and_output_and_net()
        # print(nn.compute_error())
        nn.negative_forward()

        # nn.print_w_b_list()
        # print(nn.compute_error())

        nn.positive_forward()

        # 神经网络的总误差
        network_total_error = nn.compute_error()
        if nn.iteration % 10 == 0:
            print('经过第' + str(nn.iteration) + '次迭代，网络的误差为' + str(network_total_error) + '\n')
        if network_total_error < 0.02:
            break

    # 神经网络训练完成后把误差数据传到前端
    degeneralized_input_data_list = de_generalize(nn.input_data_list, nn.input_max)
    degeneralized_output_data_list = de_generalize(nn.output_data_list, nn.output_max)
    degeneralized_actual_output_data_list = de_generalize(
        nn.layer_input_and_output_and_net[-1]['cur_layer_output_list'], nn.output_max)
    context = {
        'degeneralized_input_data_list': degeneralized_input_data_list,
        'degeneralized_output_data_list': degeneralized_output_data_list,
        'degeneralized_actual_output_data_list': degeneralized_actual_output_data_list,
        'nn_iteration': nn.iteration_arr,
        'nn_loss': nn.loss_arr
    }
    return render(request, 'training.html', context=context)


def recall(request):
    nn.positive_forward()
    degeneralized_input_data_list = de_generalize(nn.input_data_list, nn.input_max)
    degeneralized_output_data_list = de_generalize(nn.output_data_list, nn.output_max)
    degeneralized_actual_output_data_list = de_generalize(
        nn.layer_input_and_output_and_net[-1]['cur_layer_output_list'], nn.output_max)
    nn_loss = []  # 最后一次正向传播时的误差
    for i in range(len(degeneralized_output_data_list)):
        nn_loss.append(degeneralized_output_data_list[i] - degeneralized_actual_output_data_list[i])
    context = {
        'degeneralized_input_data_list': degeneralized_input_data_list,
        'degeneralized_output_data_list': degeneralized_output_data_list,
        'degeneralized_actual_output_data_list': degeneralized_actual_output_data_list,
        'nn_input_data_index': [i for i in range(len(nn.input_data_list))],
        'nn_loss': nn_loss
    }
    return render(request, 'recalling.html', context=context)


def generalize(request):
    # 用测试数据跑一遍神经网络
    nn.set_input_list(input_test_data_list)
    nn.set_input_max(input_test_max)
    nn.set_output_list(output_test_data_list)
    nn.set_output_max(output_test_max)
    nn.positive_forward()

    # 产生误差，并传向前端
    degeneralized_input_data_list = de_generalize(nn.input_data_list, nn.input_max)
    degeneralized_output_data_list = de_generalize(nn.output_data_list, nn.output_max)

    degeneralized_actual_output_data_list = de_generalize(
        nn.layer_input_and_output_and_net[-1]['cur_layer_output_list'], nn.output_max)
    nn_loss = []  # 最后一次正向传播时的误差
    for i in range(len(degeneralized_output_data_list)):
        nn_loss.append(degeneralized_output_data_list[i] - degeneralized_actual_output_data_list[i])
    context = {
        'degeneralized_input_data_list': degeneralized_input_data_list,  # 测试数据的输入
        'degeneralized_output_data_list': degeneralized_output_data_list,  # 测试数据的实际输出
        'degeneralized_actual_output_data_list': degeneralized_actual_output_data_list,  # 测试数据的理想输出
        'nn_input_data_index': [i for i in range(len(nn.input_data_list))],  # 每个测试数据的索引
        'nn_loss': nn_loss  # 每个测试数据的误差
    }
    return render(request, 'generalizing.html', context=context)