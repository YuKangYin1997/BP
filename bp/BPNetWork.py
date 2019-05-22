import math
import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pd_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def square_error(target, output):
    return 1 / 2 * np.square(target - output)


def pd_error_out(target, output):
    return output - target


class NeuronNetWork:
    LEARNING_RATE = 0.5

    def __init__(self, input_layer_neuron_num, output_layer_neuron_num, hidden_layer_neuron_num_list,
                 input_data_list, output_data_list,
                 input_max, output_max):
        """

        :param input_layer_neuron_num: 输入层的神经元的个数
        :param output_layer_neuron_num: 输出层的神经元的个数
        :param hidden_layer_neuron_num_list: 隐藏层的神经元个数的列表，表示每个隐藏层有几个神经元
        :param input_data_list: 神经网络的输入
        :param output_data_list: 神经网络的输出
        :param w_b_list: 存储所有层的w和b
        :param layer_input_and_output_and_net: 存储所有层的输入，net和out
        :param iteration: 神经网络迭代的次数
        :param input_max: 未经归一化的输入列表中的最大值
        :param output_max: 未经归一化的输出列表中的最大值
        """
        self.input_layer_neuron_num = input_layer_neuron_num
        self.output_layer_neuron_num = output_layer_neuron_num
        self.hidden_layer_neuron_num_list = hidden_layer_neuron_num_list
        self.input_data_list = input_data_list
        self.output_data_list = output_data_list
        self.w_b_list = []
        self.layer_input_and_output_and_net = []
        self.iteration = 0
        self.iteration_arr = []
        self.loss_arr = []
        self.input_max = input_max
        self.output_max = output_max

    def init_w_b(self):
        """
        w_b_list的实际长度是隐层的数目+1，因为最后一层输出层也有自己的w和b
        w_b_list的存储结构如下:[ [w1, w2, b1], [w3, w4, b2], [w5, w6, b3], # 这是一层神经网络
                               [], [], []...
                               ...
                               [], [], [] ... ]
        :return:
        """
        # 测试时使用大白话讲解中那个例子
        # self.w_b_list = [
        #     [[0.15, 0.20, 0.35], [0.25, 0.30, 0.35]],
        #     [[0.40, 0.45, 0.60], [0.50, 0.55, 0.60]]
        # ]
        for i in range(len(self.hidden_layer_neuron_num_list) + 1):
            # 对于一层神经网络
            layer_w_b_list = []
            if i < len(self.hidden_layer_neuron_num_list):  # 对于所有隐藏层
                cur_layer_neuron_num = self.hidden_layer_neuron_num_list[i]  # 该隐藏层神经网络神经元的数目
                for j in range(cur_layer_neuron_num):
                    # 对于每一个神经元，要知道他上一层神经元的数目
                    neuron_w_b_list = []
                    if i == 0:
                        # 如果是第一个隐层，那么他上一层神经元的数目就是输入层神经元的数目
                        pre_layer_neuron_num = self.input_layer_neuron_num
                    else:
                        pre_layer_neuron_num = self.hidden_layer_neuron_num_list[i - 1]
                    for k in range(pre_layer_neuron_num + 1): # 随机产生w和b
                        neuron_w_b_list.append(random.random())
                    layer_w_b_list.append(neuron_w_b_list)

            else:  # 对于最后一层输出层
                cur_layer_neuron_num = self.output_layer_neuron_num   # 输出层神经元的数目
                for j in range(cur_layer_neuron_num):
                    # 对于每一个神经元，要知道他上一层神经元的数目
                    neuron_w_b_list = []
                    pre_layer_neuron_num = self.hidden_layer_neuron_num_list[i - 1]
                    for k in range(pre_layer_neuron_num + 1):  # 随机产生w和b
                        neuron_w_b_list.append(random.random())
                    layer_w_b_list.append(neuron_w_b_list)
            # 把这一层的神经元放完了后，放到w_b_list中
            self.w_b_list.append(layer_w_b_list)

    def print_w_b_list(self):
        for i in range(len(self.w_b_list)):
            print('第' + str(i) + '层神经网络的w和b如下' + '\n')
            print('这层总共有' + str(len(self.w_b_list[i])) + '个神经元' + '\n')
            for j in range(len(self.w_b_list[i])):
                print('这层的第'+str(j)+'个神经元的w和b是')
                print(str(self.w_b_list[i][j]) + '\n')

    def layer_positive_forward(self, pre_layer_output_list, cur_layer_w_b):
        """
        单层网络的前向传播
        :param pre_layer_output_list: 前一层网络的输出列表
        :param cur_layer_w_b: 这一层网络的w和b
        :return: cur_layer_output_list 这一层的输出结果list(经过sigmoid函数)
        """
        cur_layer_input_list = pre_layer_output_list  # 存储这一层的输出数据
        cur_layer_net_list = []  # 存储这一层的net层数据
        cur_layer_output_list = []  # 存储这一层每个神经元的输出，返回给后续的层
        cur_layer_neuron_num = len(cur_layer_w_b)  # 这层网络的神经元数目
        for i in range(cur_layer_neuron_num):  # 对于这一层的每一个神经元
            w_b_for_this_neuron = cur_layer_w_b[i]  # 这个神经元的w_b
            net = 0  # 这个神经元的net
            for j in range(len(cur_layer_input_list)):
                net += cur_layer_input_list[j] * w_b_for_this_neuron[j]  # w与输入相乘
            net += w_b_for_this_neuron[-1]  # 加上这个神经元的b值
            cur_layer_net_list.append(net)
            out = sigmoid(net)
            cur_layer_output_list.append(out)

        # 把这一层的输入,net,out都存储起来
        cur_layer_input_and_output_and_net = {
            'cur_layer_input_list': cur_layer_input_list,
            'cur_layer_net_list': cur_layer_net_list,
            'cur_layer_output_list': cur_layer_output_list
        }
        self.layer_input_and_output_and_net.append(cur_layer_input_and_output_and_net)
        return cur_layer_output_list

    def positive_forward(self):
        """
        这个神经网络的正向传播
        :return:
        """
        pre_layer_output_list = []
        for i in range(len(self.hidden_layer_neuron_num_list) + 1):  # 一共有隐藏层数目+1次传播
            if i == 0:  # 如果是第一个隐藏层，需要特殊处理，因为他的输入是神经网络的输入
                pre_layer_output_list = self.input_data_list

            cur_layer_w_b = self.w_b_list[i]

            # 单层进行前向传播，得到的结果是这一层的输出，也就是下一层的输入
            cur_layer_output_list = self.layer_positive_forward(pre_layer_output_list=pre_layer_output_list,
                                                                cur_layer_w_b=cur_layer_w_b)
            # 把这一层的输出传给下一层，进行下一个循环
            pre_layer_output_list = cur_layer_output_list

    def print_layer_input_and_output_and_net(self):
        for i in range(len(self.layer_input_and_output_and_net)):
            print('\n第'+str(i)+'层的神经网络的输入,net,out如下' + '\n')
            print('input:')
            print(self.layer_input_and_output_and_net[i]['cur_layer_input_list'])
            print('\nnet:')
            print(self.layer_input_and_output_and_net[i]['cur_layer_net_list'])
            print('\noutput:')
            print(self.layer_input_and_output_and_net[i]['cur_layer_output_list'])

    def compute_error(self):
        """
        根据输出层的输出和神经网络期望的输出，求出经过一遍正向传播后神经网络的总误差
        :return:
        """
        error_num = 0
        for i in range(len(self.output_data_list)):
            target = self.output_data_list[i]
            output = self.layer_input_and_output_and_net[-1]['cur_layer_output_list'][i]
            error_num += square_error(target=target, output=output)

        # 把当前的迭代次数放入迭代次数的数组中
        self.iteration_arr.append(self.iteration)
        # 把当前的误差放入误差的数组中
        self.loss_arr.append(error_num)
        # 加入画图函数
        if self.iteration % 10 == 0:
            # 对输入数据，理想输出数据，实际输出数据进行反归一化
            degeneralized_input_data_list = de_generalize(self.input_data_list, self.input_max)
            degeneralized_output_data_list = de_generalize(self.output_data_list, self.output_max)
            degeneralized_actual_output_data_list = de_generalize(self.layer_input_and_output_and_net[-1]['cur_layer_output_list'], self.output_max)
            # draw_fit_curve(origin_xs=self.input_data_list, target_ys=self.output_data_list,
            #                actual_ys=self.layer_input_and_output_and_net[-1]['cur_layer_output_list'],
            #                step_arr=self.iteration_arr, loss_arr=self.loss_arr)
            draw_fit_curve(origin_xs=degeneralized_input_data_list, target_ys=degeneralized_output_data_list,
                           actual_ys=degeneralized_actual_output_data_list,
                           step_arr=self.iteration_arr, loss_arr=self.loss_arr)
        return error_num

    def layer_negative_forward(self, pd_totalerror_pre_layer_net_list, pre_layer_updated_w_b_list, cur_layer_input_list,
                               cur_layer_number):
        """

        :param pd_totalerror_pre_layer_net_list: 总误差对前一层中每个神经元的net层的偏导数组成的list
        :param pre_layer_updated_w_b_list: 前一层更新过后的w_b的list
        :param cur_layer_input_list: 这一层的输入的list（注意是input不是net）
        :param cur_layer_number: 这一层在神经网络中的序号（第一个隐层为0层，最后一个输出层为最后一层）
        :return: pd_totalerror_cur_layer_net_list 总的误差对这一层所有神经元的net层的偏导数
        """
        debug = 1
        pd_totalerror_cur_layer_net_list = []
        need_to_be_updated_neuron_num = len(pre_layer_updated_w_b_list[0]) - 1
        for i in range(need_to_be_updated_neuron_num): # 对于这一层每一个神经元
            sum = 0  # 存储总误差对这个神经元net层的偏导数
            for j in range(len(pd_totalerror_pre_layer_net_list)):  # 总误差对前一层每个神经元的net层的偏导数
                sum += pd_totalerror_pre_layer_net_list[j] * pre_layer_updated_w_b_list[j][i]
            # 到这里为止是把总误差对这一层的一个神经元的net层的偏导数准备好了，现在要把它放进一个list中，给再前面一层传递
            pd_totalerror_cur_layer_net_list.append(sum)

            # 对这个神经元的w和b进行更新
            # 这一个神经元的输出,由它的层数找到这一层的输出列表，在由这个神经元的序号找到它的输出的值
            neuron_output = self.layer_input_and_output_and_net[cur_layer_number]['cur_layer_output_list'][i]
            for m in range(len(cur_layer_input_list)):  # 这个神经元的所有w
                pd = sum * (neuron_output * (1 - neuron_output)) * cur_layer_input_list[m] * self.LEARNING_RATE
                # 计算除了w的偏导数，更新w
                self.w_b_list[cur_layer_number][i][m] -= pd  # 有错误
            # 更新这个神经元的一个b
            self.w_b_list[cur_layer_number][-1] -= sum * (neuron_output * (1 - neuron_output)) * self.LEARNING_RATE
        return pd_totalerror_cur_layer_net_list


    def negative_forward(self):
        """
        整个神经网络的反向传播
        :return:
        """
        pd_totalerror_cur_layer_net_list = []  # 总误差对这一层的net层的偏导数组成的list
        total_layer_num = len(self.hidden_layer_neuron_num_list)  # 加上输出层

        for i in range(total_layer_num, -1, -1):
            if i == total_layer_num:  # 特殊处理开始时的输出层
                for j in range(len(self.output_data_list)): # 对于输出层的每一个神经元
                    sum = 0  # 存储总误差对这个神经元net层的偏导数
                    # 这一个神经元的输出,由它的层数找到这一层的输出列表，在由这个神经元的序号找到它的输出的值
                    neuron_output = self.layer_input_and_output_and_net[-1]['cur_layer_output_list'][j]
                    target_output = self.output_data_list[j]
                    sum += pd_error_out(target=target_output, output=neuron_output) * (neuron_output * (1 - neuron_output))
                    # 把总误差对这个神经元的偏导数放入总误差对这一层的net层的偏导数组成的list中
                    pd_totalerror_cur_layer_net_list.append(sum)

                    # 找到和这个神经元相连的权值和b
                    output_layer_w_b = self.w_b_list[-1][j]
                    for m in range(len(output_layer_w_b)):
                        if m == len(output_layer_w_b) - 1: # 单独处理b
                            self.w_b_list[-1][j][-1] -= sum * self.LEARNING_RATE  # 最后一个w_b_list的第j个神经元的最后一个就是b
                        else:  # 处理w
                            # 这个神经元的每个输入项
                            for n in range(len(self.layer_input_and_output_and_net[-1]['cur_layer_input_list'])):
                                self.w_b_list[-1][j][m] -= \
                                    sum * self.layer_input_and_output_and_net[-1]['cur_layer_input_list'][n]* self.LEARNING_RATE
            else:
                pd_totalerror_cur_layer_net_list = self.layer_negative_forward(pd_totalerror_pre_layer_net_list=pd_totalerror_cur_layer_net_list,
                                            pre_layer_updated_w_b_list=self.w_b_list[i + 1],
                                            cur_layer_input_list=self.layer_input_and_output_and_net[i]['cur_layer_input_list'],
                                            cur_layer_number=i)

    def set_input_list(self, input_data_list):
        self.input_data_list = input_data_list

    def set_output_list(self, output_data_list):
        self.output_data_list = output_data_list

    def set_input_max(self, input_max):
        self.input_max = input_max

    def set_output_max(self, output_max):
        self.output_max


def generate_input_and_output_data(input_begin, input_end, function_name):
    output_data_list = []
    input_data_list = list(np.arange(input_begin, input_end, 0.01))
    for i in range(len(input_data_list)):
        if function_name is 'sin':
            output_data_list.append(math.sin(input_data_list[i]) + 1)  # y = sin(x) + 1
        elif function_name is 'xsquare':
            output_data_list.append(math.pow(input_data_list[i], 2))  # y = x * x
    return input_data_list, output_data_list


# 归一化
def generalize(input_data_list, output_data_list):
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


if __name__ == '__main__':
    # 随机产生输入和输出，是没有归一化的数据
    ori_input_data_list, ori_output_data_list = generate_input_and_output_data(-5, 5, 'sin')

    # print('归一化之前:')
    # print('没有归一化的input_datas:' + str(ori_input_data_list))
    # print('没有归一化的output_datas:' + str(ori_output_data_list))

    # 归一化
    # input_max为input_data_list中的最大值,output_max为output_data_list中的最大值
    # global input_max, output_max
    input_data_list, output_data_list, input_max, output_max = generalize(input_data_list=ori_input_data_list,
                                                                          output_data_list=ori_output_data_list)

    # print('归一化之后:')
    # print(input_data_list)
    # print(output_data_list)
    #
    # print('反归一化后:')
    # print(de_generalize(input_data_list, input_max))
    # print(de_generalize(output_data_list, output_max))

    # 创建神经网络
    nn = NeuronNetWork(input_layer_neuron_num=len(input_data_list), output_layer_neuron_num=len(input_data_list),
                       hidden_layer_neuron_num_list=[3],
                       input_data_list=input_data_list, output_data_list=output_data_list,
                       input_max=input_max, output_max=output_max)

    # 初始化神经网络的w和b
    nn.init_w_b()
    # nn.print_w_b_list()

    index = 0  # 迭代次数
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
        if network_total_error < 0.2:
            break

    # 神经网络训练完成后，进行回想测试，把原来输入的那些测试数据再次输入神经网络，看误差
    nn.positive_forward()
    print('神经网络训练完成后，再次输入测试数据后的误差是:' + str(nn.compute_error()))

    # 神经网络训练完成后，还需要看看测试数据的误差

