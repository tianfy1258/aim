import numpy as np
import pandas as pd

from aim.utils import *
import random as r
from aim.utils import parse_model
import torch


class BaseCoverage:
    """
    基类覆盖对象，实现了通用方法
    具体归因算法，只需要实现coverage方法即可，案例见下方NeuronCoverage类
    """

    def __init__(self, req: dict, coverage_model):
        self.model_id = req['model_id']
        self.options = req['options']
        self.coverage_method = req['coverage']
        self.dataset_id = req['dataset_id']
        self.sample_count = req['sample_count']
        self.threshold = req['threshold']
        self.max_threshold = req['max_threshold']
        self.min_threshold = req['min_threshold']
        # 加载模型信息
        deep_model = DeepModel.objects.get(model_id=self.model_id)
        # 获取数据集信息
        self.dataset = Dataset.objects.get(dataset_id=self.dataset_id)
        # 加载数据集标签信息
        self.dataset_df = pd.read_csv(fr"{DATA_PATH}\{self.dataset.dataset_filename}.csv")
        self.dataset_df = self.dataset_df.set_index("filename")
        # 加载模型，net, func, classes
        self.model, self.func, self.classes = parse_model(deep_model.model_filename,
                                                          deep_model.model_classname,
                                                          deep_model.model_processor,
                                                          is_use_function=deep_model.is_use_function
                                                          )
        self.coverage_model = coverage_model(self.model)
        self.interrupted = False
        self.timer = time.time()
        self.result = []

    def coverage(self, tensor_image: torch.Tensor) -> Tuple[int, int, float]:
        """
        进行覆盖，返回覆盖结果
        :param tensor_image: 输入图片，torch.Tensor格式
        :return: 覆盖数，总数，覆盖率
        """
        # 不是所有参数都有用，但是为了保持接口一致，这里保留了所有参数
        # 某个模型只会用到特定的某个参数
        self.coverage_model.update_coverage(tensor_image,
                                            threshold=self.threshold,
                                            min_threshold=self.min_threshold,
                                            max_threshold=self.max_threshold)
        return  self.coverage_model.neuron_coverage_rate()

    def is_finished(self):
        return len(self.result) == self.sample_count

    def interrupt(self):
        self.interrupted = True

    # 对外接口，用于调用
    def coverage_from_dataset(self):
        # 从缓存中获取zip数据集文件
        zfile = get_zipfile(self.dataset.dataset_filename)

        # 根据采样数量获得要分析的图片samples
        filenames = [x.filename for x in zfile.filelist]
        samples = r.sample(filenames, k=self.sample_count)

        # 将结果保存至self.result
        for filename, image in zip(samples, extract_images(zfile, samples, use_cache=False)):
            # 被打断或者超过5秒未获取结果，停止计算
            if self.interrupted or time.time() - self.timer > 5:
                break
            # filename: str
            # image: PIL.Image
            # 获取numpy格式图片
            image_arr: np.ndarray = np.array(image)
            # 使用模型的processor对tensor进行处理，得到tensor格式的处理后图片
            tensor_image: torch.Tensor = self.func(image_arr)
            # 如果不是4维，多加一个Batch维度
            if len(tensor_image.shape) == 3:
                tensor_image = tensor_image.unsqueeze(0)
            res = self.coverage(tensor_image)
            self.result.append(res)
            # LOGGER.debug(f"res： {res}")

        return self.result[-1]

    def get_result(self):
        self.timer = time.time()
        return self.result
