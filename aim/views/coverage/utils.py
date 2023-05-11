import torch
import numpy as np

from aim.utils import LOGGER


class NC:
    def __init__(self, model):
        self.model = model.eval()
        self.all_layer_name = self._get_all_layer_name()
        self.model_layer_dict = dict()
        self.all_out = []
        # register hooks
        for name, modules in self.model.named_modules():
            if name in self.all_layer_name:
                modules.register_forward_hook(lambda module, input, output: self.all_out.append(output))

    def _get_all_layer_name(self):
        all_layer_name = []
        layers = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear]
        for name, m in self.model.named_modules():
            isValid = len([1 for layer in layers if isinstance(m, layer)]) > 0
            if isValid:
                all_layer_name.append(name)

        LOGGER.info(all_layer_name)
        return all_layer_name

    def update_coverage(self, data, **kwargs):
        threshold = kwargs.get('threshold', 0.75)
        layer_name = self.all_layer_name
        # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)
        # batch_size = 1
        self.model.zero_grad()
        intermediate_layer_outputs = self._get_forward_value(data)
        for i, output in enumerate(intermediate_layer_outputs):
            # output 1,c,h,w
            scaled = self._scale(output[0])
            for num_neuron in range(scaled.shape[0]):  # 遍历c
                if (layer_name[i], num_neuron) not in self.model_layer_dict:
                    self.model_layer_dict[(layer_name[i], num_neuron)] = False
                if np.mean(scaled[num_neuron]) > threshold \
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
        with torch.no_grad():
            _ = self.model(data)
        return self.all_out


class NBC:
    def __init__(self, model):
        self.model = model.eval()
        self.all_layer_name = self._get_all_layer_name()
        self.model_layer_dict = dict()
        self.all_out = []
        # register hooks
        for name, modules in self.model.named_modules():
            if name in self.all_layer_name:
                modules.register_forward_hook(lambda module, input, output: self.all_out.append(output))

    def _get_all_layer_name(self):
        all_layer_name = []
        layers = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear]
        for name, m in self.model.named_modules():
            isValid = len([1 for layer in layers if isinstance(m, layer)]) > 0
            if isValid:
                all_layer_name.append(name)

        LOGGER.info(all_layer_name)
        return all_layer_name

    def update_coverage(self, data, **kwargs):
        min_threshold = kwargs.get('min_threshold', 0.03)
        max_threshold = kwargs.get('max_threshold', 0.97)
        layer_name = self.all_layer_name
        # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)
        # batch_size = 1
        self.model.zero_grad()
        intermediate_layer_outputs = self._get_forward_value(data)
        for i, output in enumerate(intermediate_layer_outputs):
            # output 1,c,h,w
            scaled = output[0].cpu().numpy()
            for num_neuron in range(scaled.shape[0]):  # 遍历c
                if (layer_name[i], num_neuron) not in self.model_layer_dict:
                    self.model_layer_dict[(layer_name[i], num_neuron)] = False
                if (np.mean(scaled[num_neuron]) > max_threshold or np.mean(scaled[num_neuron]) < min_threshold) \
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
        with torch.no_grad():
            _ = self.model(data)
        return self.all_out


class TKNC:
    def __init__(self, model):
        self.model = model.eval()
        self.all_layer_name = self._get_all_layer_name()
        self.model_layer_dict = dict()
        self.all_out = []
        # register hooks
        for name, modules in self.model.named_modules():
            if name in self.all_layer_name:
                modules.register_forward_hook(lambda module, input, output: self.all_out.append(output))

    def _get_all_layer_name(self):
        all_layer_name = []
        layers = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear]
        for name, m in self.model.named_modules():
            isValid = len([1 for layer in layers if isinstance(m, layer)]) > 0
            if isValid:
                all_layer_name.append(name)

        LOGGER.info(all_layer_name)
        return all_layer_name

    def update_coverage(self, data, **kwargs):
        threshold = kwargs.get('threshold', 0.75)
        layer_name = self.all_layer_name
        # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)
        # batch_size = 1
        self.model.zero_grad()
        intermediate_layer_outputs = self._get_forward_value(data)
        for i, output in enumerate(intermediate_layer_outputs):
            # output 1,c,h,w or 1,n
            topk_idx = None
            scaled = self._scale(output[0])
            k = int(threshold * scaled.shape[0])
            if len(output.shape) == 4:
                # (c,h,w) -> (c,1)
                avg_arr = np.mean(scaled, axis=(1, 2))
                topk_idx = np.argsort(avg_arr)[::-1][:k]
            elif len(output.shape) == 2:
                # (n,)
                topk_idx = np.argsort(scaled)[::-1][:k]

            for num_neuron in range(scaled.shape[0]):  # 遍历c
                if (layer_name[i], num_neuron) not in self.model_layer_dict:
                    self.model_layer_dict[(layer_name[i], num_neuron)] = False
                if num_neuron in topk_idx \
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
        with torch.no_grad():
            _ = self.model(data)
        return self.all_out

class SNAC:
    def __init__(self, model):
        self.model = model.eval()
        self.all_layer_name = self._get_all_layer_name()
        self.model_layer_dict = dict()
        self.all_out = []
        # register hooks
        for name, modules in self.model.named_modules():
            if name in self.all_layer_name:
                modules.register_forward_hook(lambda module, input, output: self.all_out.append(output))

    def _get_all_layer_name(self):
        all_layer_name = []
        layers = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear]
        for name, m in self.model.named_modules():
            isValid = len([1 for layer in layers if isinstance(m, layer)]) > 0
            if isValid:
                all_layer_name.append(name)

        LOGGER.info(all_layer_name)
        return all_layer_name

    def update_coverage(self, data, **kwargs):
        min_threshold = kwargs.get('min_threshold', 0.03)
        max_threshold = kwargs.get('max_threshold', 0.97)
        layer_name = self.all_layer_name
        # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)
        # batch_size = 1
        self.model.zero_grad()
        intermediate_layer_outputs = self._get_forward_value(data)
        for i, output in enumerate(intermediate_layer_outputs):
            # output 1,c,h,w
            scaled = output[0].cpu().numpy()
            for num_neuron in range(scaled.shape[0]):  # 遍历c
                if (layer_name[i], num_neuron) not in self.model_layer_dict:
                    self.model_layer_dict[(layer_name[i], num_neuron)] = False
                if np.mean(scaled[num_neuron]) > max_threshold \
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
        with torch.no_grad():
            _ = self.model(data)
        return self.all_out
# import torch
# import numpy as np
#
# from aim.utils import LOGGER
#
#
# class NC:
#     def __init__(self, model):
#         self.model = model.eval()
#         self.all_layer_name = self._get_all_layer_name()
#         self.model_layer_dict = dict()
#         self.all_out = []
#         # register hooks
#         for name, modules in self.model.named_modules():
#             if name in self.all_layer_name:
#                 modules.register_forward_hook(lambda module, input, output: self.all_out.append(output))
#
#     def _get_all_layer_name(self):
#         all_layer_name = []
#         layers = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear]
#         for name, m in self.model.named_modules():
#             isValid = len([1 for layer in layers if isinstance(m, layer)]) > 0
#             if isValid:
#                 all_layer_name.append(name)
#
#         LOGGER.info(all_layer_name)
#         return all_layer_name
#
#     def update_coverage(self, data, threshold=0.75):
#         layer_name = self.all_layer_name
#         # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)
#         # batch_size = 1
#         self.model.zero_grad()
#         intermediate_layer_outputs = self._get_forward_value(data)
#         for i, output in enumerate(intermediate_layer_outputs):
#             # 卷积层
#             if len(output.shape) == 4:
#                 # output 1,c,h,w -> c,h,w
#                 output = output[0]  # 删除batch维度
#
#             # 全连接层
#             elif len(output.shape) == 2:
#                 output = output.unsqueeze(0)
#
#             for channel_index in range(output.shape[0]):  # 遍历特征图
#                 channel = output[channel_index]
#                 # scaled (h,w)
#                 scaled = self._scale(channel)
#                 activated_neurons = scaled > threshold
#                 key = (layer_name[i], channel_index)
#
#                 # Update the model_layer_dict with accumulated coverage
#                 if key in self.model_layer_dict:
#                     prev_activated_neurons, _, total_neurons = self.model_layer_dict[key]
#                     combined_activated_neurons = np.logical_or(prev_activated_neurons, activated_neurons)
#                     num_activated_neurons = np.sum(combined_activated_neurons)
#                     self.model_layer_dict[key] = (combined_activated_neurons, num_activated_neurons, total_neurons)
#                 else:
#                     total_neurons = channel.shape[0] * channel.shape[1]
#                     self.model_layer_dict[key] = (activated_neurons, np.sum(activated_neurons), total_neurons)
#
#     def _scale(self, input, rmax=1, rmin=0):
#         input = input.cpu().detach().numpy()
#         input_std = (input - np.min(input)) / (input.max() - input.min())
#         input_scaled = input_std * (rmax - rmin) + rmin
#         return input_scaled
#
#     # 按照图片的行进行归一化 wrong
#     # def _scale(self,out, dim=-1, rmax=1, rmin=0):
#     #     out_max = out.max(dim)[0].unsqueeze(dim)
#     #     out_min = out.min(dim)[0].unsqueeze(dim)
#     #     '''
#     #         out_max = out.max()
#     #         out_min = out.min()
#     #     Note that the above max/min is incorrect when batch_size > 1
#     #     '''
#     #     output_std = (out - out_min) / (out_max - out_min)
#     #     output_scaled = output_std * (rmax - rmin) + rmin
#     #     return output_scaled
#
#     def neuron_coverage_rate(self):
#         covered_neurons = 0
#         total_neurons = 0
#
#         # 遍历 model_layer_dict 中的每个值 (activated_neurons, num_activated_neurons, total_neurons_in_channel) 对
#         for layer_key, neuron_count in self.model_layer_dict.items():
#             _, num_activated_neurons, total_neurons_in_channel = neuron_count
#             covered_neurons += num_activated_neurons
#             total_neurons += total_neurons_in_channel
#
#         coverage_rate = covered_neurons / float(total_neurons)
#
#         return covered_neurons, total_neurons, coverage_rate
#
#     def _get_forward_value(self, data):
#         self.all_out = []
#         with torch.no_grad():
#             _ = self.model(data)
#         return self.all_out
