import torch
import numpy as np


class NC:
    def __init__(self, model):
        self.model = model
        self.all_layer_name = self._get_all_layer_name()
        self.model_layer_dict = dict()
        self.all_out = []
        # register hooks
        for name, modules in self.model.named_modules():
            if name in self.all_layer_name:

                modules.register_forward_hook(lambda module,input,output: self.all_out.append(output))

    def _get_all_layer_name(self):
        all_layer_name = []
        layers = [torch.nn.MaxPool2d]
        for name, m in self.model.named_modules():
            isValid = len([1 for layer in layers if isinstance(m, layer)]) > 0
            if isValid:
                all_layer_name.append(name)
        return all_layer_name

    def update_coverage(self, data, threshold=0.75):
        layer_name = self.all_layer_name
        # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)
        # batch_size = 1
        intermediate_layer_outputs = self._get_forward_value(data)
        for i, output in enumerate(intermediate_layer_outputs):
            # output 1,c,h,w
            scaled = self._scale(output[0])
            for num_neuron in range(scaled.shape[0]):  # 遍历c
                if (layer_name[i], num_neuron) not in self.model_layer_dict:
                    self.model_layer_dict[(layer_name[i], num_neuron)] = False
                if np.mean(scaled[num_neuron, ...]) > threshold \
                        and not self.model_layer_dict.get((layer_name[i], num_neuron)):
                    self.model_layer_dict[(layer_name[i], num_neuron)] = True

    def _scale(self, input, rmax=1, rmin=0):
        input = input.cpu().detach().numpy()
        input_std = (input - np.min(input)) / (input.max() - input.min())
        input_scaled = input_std * (rmax - rmin) + rmin
        return input_scaled

    def neuron_coverage_rate(self):
        # 统计神经元覆盖率
        covered_neurons = len([v for v in self.model_layer_dict.values() if v])
        total_neurons = len(self.model_layer_dict)

        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def _get_forward_value(self, data):
        self.all_out = []
        _ = self.model(data)
        return self.all_out
