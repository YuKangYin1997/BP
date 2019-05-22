import random
import math
import numpy as np


#
#   参数解释：
#   "pd_" ：偏导的前缀
#   "d_" ：导数的前缀
#   "w_ho" ：隐含层到输出层的权重系数索引
#   "w_ih" ：输入层到隐含层的权重系数的索引
class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, hidden_layer_bias,
                 output_layer_weights, output_layer_bias, inputs):
        # 初始化输入层的结点的个数
        self.num_inputs = num_inputs

        # 初始化隐含层，参数为隐含层神经元的数目和隐含层的截距bias
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias, inputs=inputs, weight=hidden_layer_weights)

        # 初始化输出层，参数为输出层神经元的数目和输出层的截距bias
        # 输出层的输入参数是隐含层的输出参数，所以需要让隐含层先进行一下前向传播
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias, inputs=self.hidden_layer.feed_forward(),
                                        weight=output_layer_weights)
        self.output_layer.feed_forward()

        # 初始化输入层到隐含层的权值w1~w4
        # self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)

        # 初始化隐含层到输出层的权值w5~w8
        # self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
    #     weight_num = 0
    #     for h in range(len(self.hidden_layer.neurons)):  # 对于隐含层的每个神经元
    #         for i in range(self.num_inputs):  # 对于输入层的每个输入神经元
    #             if not hidden_layer_weights:
    #                 self.hidden_layer.neurons[h].weights.append(random.random())
    #             else:
    #                 self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
    #             weight_num += 1

    # def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
    #     weight_num = 0
    #     for o in range(len(self.output_layer.neurons)):  # 对于输出层的每个神经元
    #         for h in range(len(self.hidden_layer.neurons)):  # 对于隐含层的每个神经元
    #             if not output_layer_weights:
    #                 self.output_layer.neurons[o].weights.append(random.random())
    #             else:
    #                 self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
    #             weight_num += 1

    # def inspect(self):
    #     print('------')
    #     print('* Inputs: {}'.format(self.num_inputs))
    #     print('------')
    #     print('Hidden Layer')
    #     self.hidden_layer.inspect()
    #     print('------')
    #     print('* Output Layer')
    #     self.output_layer.inspect()
    #     print('------')

    def feed_forward(self):
        """

        :param inputs: 输入层输入的数据
        :return:
        """
        # 隐含层拿到输入层输入的数据，计算出结果
        hidden_layer_outputs = self.hidden_layer.feed_forward()
        # 隐含层把结果作为输出层的输入
        return self.output_layer.feed_forward()

    def train(self, training_inputs, training_outputs):
        """

        :param training_inputs: 输入层输入的数据
        :param training_outputs: 期望输出层输出的数据
        :return:
        """
        # 所有层把输入的数据都向前面一层传播，完成了一遍向前传播，在输出层产生了误差
        # self.feed_forward(training_inputs)
        # 1. 输出神经元的值
        # 单个输出层的误差与输出层的输入之间的偏导数
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ 计算输出层的误差与输出层的输入之间的偏导数
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[
                o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
        # 2. 隐含层神经元的值
        # 两个输出层的误差与隐含层的输出之间的偏导数
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * \
                                                    self.output_layer.neurons[o].weights[h]
            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * \
                                                             self.hidden_layer.neurons[
                                                                 h].calculate_pd_total_net_input_wrt_input()
        # 3. 更新输出层权重系数
        for o in range(len(self.output_layer.neurons)):
            # 隐含层到输出层
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                # 单个输出层的误差与输出层的输入之间的偏导数
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[
                    o].calculate_pd_total_net_input_wrt_weight(w_ho)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
        # 4. 更新隐含层的权重系数
        for h in range(len(self.hidden_layer.neurons)):
            # 输入层到隐含层
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                # 两个输出层的误差与隐含层的输出之间的偏导数 * 隐含层的输出与隐含层的输入之间的偏导数
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[
                    h].calculate_pd_total_net_input_wrt_weight(w_ih)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            # 训练完之后，重新跑一遍输入的数据
            self.feed_forward()
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons, bias, inputs, weight):
        print("这一层神经网络的输入是" + str(inputs) + "这一层神经网络的weight是" + str(weight))
        # 同一层的神经元共享一个截距项b
        self.bias = bias if bias else random.random()
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias, weight=[weight[i * num_neurons], weight[i * num_neurons + 1]],
                                       inputs=inputs))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self):
        outputs = []
        for neuron in self.neurons:
            # 这一层的输出列表中加入每一个神经元的输出
            outputs.append(neuron.calculate_output())
        print("这一层神经网络的输出是" + str(outputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias, weight, inputs):
        # bias是截距
        self.bias = bias
        self.weights = weight
        self.inputs = inputs
        self.output = 0
        # print('这个神经元的输入是' + str(inputs) + "这个神经元的weight是" + str(weight))

    # 所有输入值的和通活化函数得到输出值
    def calculate_output(self):
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output

    # 计算所有输入值的和
    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.weights)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # 激活函数sigmoid
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # 输出层的误差与输出层的输入之间的偏导数
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

    # 每一个神经元的误差是由平方差公式计算的
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # 输出层的误差与输出层的输出的偏导数
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # 输出层的输出与输出层的输入之间的偏导数
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


if __name__ == '__main__':
    # h1 = Neuron(bias=0.35, weight=[0.15, 0.25], inputs=[0.05, 0.10])
    # print(h1.calculate_output())
    # layer1 = NeuronLayer(num_neurons=2, bias=0.35, inputs=[0.05, 0.10], weight=[0.15, 0.20, 0.25, 0.30])
    # output = layer1.feed_forward()
    # print(output)
    nn = NeuralNetwork(num_inputs=2, num_hidden=2, num_outputs=2,
                       hidden_layer_weights=[0.15, 0.20, 0.25, 0.30], hidden_layer_bias=0.35,
                       output_layer_weights=[0.40, 0.45, 0.50, 0.55], output_layer_bias=0.60,
                       inputs=[3.141592653589793, 4.71238898038469])
    list_input = [3.141592653589793, 4.71238898038469]
    list_output = [0.0, -1.0]
    for i in range(10000):
        nn.train(list_input, list_output)
        # nn.inspect()
        # 训练完之后, 小数点后保留9位
        # print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))
        total_error = round(nn.calculate_total_error([[list_input, list_output]]), 9)

        # total_errors.append(round(nn.calculate_total_error([[list_input, list_output]]), 9))
        print(i, total_error)
    # hidden_layer_weights和output_layer_weights不能体现在参数中
