import numpy as np
import pandas as pd

from aim.utils import *
import random as r
from aim.utils import parse_model
import torch
from captum.attr import visualization as viz

IMAGE_CACHE = ImageFileCache()
ATTRIBUTION_CACHE = AttributionCache()
ATTRIBUTE_RESULT_CACHE = AttributeResultCache()


class BaseAttribution:
    """
    基类归因对象，实现了通用方法
    具体归因算法，只需要实现attribute方法即可，案例见下方IntegratedGradients类
    """

    def __init__(self, req: dict):
        self.model_id = req['model_id']
        self.options = req['options']
        self.is_noise_tunnel = req['is_noise_tunnel']
        self.noise_tunnel_options = req['noise_tunnel_options']
        self.visualize = req['visualize']
        self.sign = req['sign']
        self.attribution = req['attribution']
        self.dataset_id = req['dataset_id']
        self.sample_method = req['sample_method']
        self.sample_num = req['sample_num']
        self.samples = req['samples'] if "samples" in req else None
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

    # 对内接口，不对外
    def attribute(self, tensor_image: torch.Tensor, label: int) -> np.ndarray:
        """
        主要实现这个方法
        输入一个(H,W,C)的numpy矩阵，输出一个(H,W,C)的numpy矩阵
        """
        pass

    # 对内接口，不对外
    def attribute_image_features(self, algorithm, input: torch.Tensor, target: int, **kwargs):
        self.model.zero_grad()
        input.requires_grad_(True)
        if self.is_noise_tunnel:
            from captum.attr import NoiseTunnel
            algorithm = NoiseTunnel(algorithm)
            tensor_attributions = algorithm.attribute(
                input,
                target=target,
                nt_type=self.noise_tunnel_options["nt_type"],
                nt_samples=self.noise_tunnel_options["nt_samples"],
                stdevs=float(self.noise_tunnel_options["stdevs"]),
            )

        else:
            tensor_attributions = algorithm.attribute(input,
                                                      target=target,
                                                      **kwargs
                                                      )

        return tensor_attributions

    # 对外接口，用于调用
    def attribute_from_dataset(self):

        # 从缓存中获取zip数据集文件
        zfile = get_zipfile(self.dataset.dataset_filename)

        # 根据采样方法获得要分析的图片samples
        samples = None
        # 随机采样
        if self.sample_method == "random":
            filenames = [x.filename for x in zfile.filelist]
            samples = r.sample(filenames, k=self.sample_num)
        # 前端提供之前结果的图片列表
        elif self.sample_method == "provide":
            samples = self.samples
        # 前端提供自己选择的图片列表（待实现）
        elif self.sample_method == "custom":
            pass

        # 将结果保存至result
        result = []
        for filename, image in zip(samples, extract_images(zfile, samples)):
            # filename: str
            # image: PIL.Image

            # 获取numpy格式图片
            image_arr: np.ndarray = np.array(image)
            # 使用模型的processor对tensor进行处理，得到tensor格式的处理后图片
            tensor_image: torch.Tensor = self.func(image_arr)
            # 对原图片进行resize
            image_arr = np.array(image.resize(tensor_image.shape[1:]))
            # 如果不是4维，多加一个Batch维度
            if len(tensor_image.shape) == 3:
                tensor_image = tensor_image.unsqueeze(0)
            # 获得模型对该图片的分类结果（int值）
            # (1,C,H,W) -> (1,output_dim)
            with torch.no_grad():
                _outputs: torch.Tensor = self.model(tensor_image)
            if not isinstance(_outputs, torch.Tensor):
                raise CustomException(f"模型输出为{type(_outputs)}，应该为torch.Tensor")
            if not (_outputs.shape[0] == 1 and len(_outputs.shape) == 2):
                raise CustomException(f"模型输出为大小为{tuple(_outputs.shape)}，应该为(1,标签个数)")
            # (1,output_dim) -> int
            _, output = torch.max(_outputs, 1)
            index = output.item()
            score = torch.softmax(_outputs, 1)[0, index].item()
            # 获得模型对该图片的分类
            predict_label = self.classes[index]
            # 获得图片本身的分类
            label = self.dataset_df.loc[filename][0]
            # 计算归因结果并写入缓存

            key = md5(
                f"{self.model_id}"
                f"{self.dataset_id}"
                f"{self.options}"
                f"{self.is_noise_tunnel}"
                f"{self.noise_tunnel_options}"
                f"{filename}"
                f"{self.attribution}"
            )
            a = time.time()
            attr: np.ndarray = ATTRIBUTE_RESULT_CACHE.get(key,None)
            if attr is None:
                # 获取numpy格式的归因结果
                attr: np.ndarray = self.attribute(tensor_image, index)
                ATTRIBUTE_RESULT_CACHE.set(key,attr)
            LOGGER.debug(f"归因耗时：{time.time() - a}s")

            # 对归因结果attr进行维度处理 (1,C,H,W) -> (H,W,C)
            attr = attr.squeeze().permute(1, 2, 0).detach().numpy()
            # 对结果可视化，将归因结果处理为PIL.Image格式的attr_image
            fig = viz.visualize_image_attr(attr, image_arr, method=self.visualize, sign=self.sign,
                                           show_colorbar=True, title="", use_pyplot=False)
            fig = fig[0]
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            attr_image: Image = Image.open(buf)
            # 终归我还是败了，在N小时的debug后，下面这种性能更高的方法，captum用不了，只能用savefig的方式写入内存
            # 因为captum某些代码和FigureCanvasBase有一腿，也就是说不能用agg做后台，所以tostring_rgb用不了
            # 直接传入用agg做后台的Figure对象出不来图片，原因未知
            # 如果能用，输出每张图片性能提升约为100ms~200ms
            # setattr(fig.canvas, "renderer", fig.canvas.get_renderer())
            # attr_image = Image.frombytes('RGB',
            #                     fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

            # 写入文件名，置信度，分类类别，图片类别，图片，归因后的输出图片
            result.append((filename, score, predict_label, label, image, attr_image,))

        # 将图片和归因后的输出图片存入缓存，并返回url
        result = [
            {
                "image_name": filename,
                "image_score": score,
                "image_predict_label": predict_label,
                "image_label": label,
                "image_url": IMAGE_CACHE.set(f"i{filename}", image),
                "attr_url": IMAGE_CACHE.set(f"o{filename}", attr_image),
            }
            for filename, score, predict_label, label, image, attr_image in result
        ]
        return result


class IntegratedGradients(BaseAttribution):
    def __init__(self, req: dict):
        super().__init__(req)

    def attribute(self, tensor_image: torch.Tensor, label: int) -> np.ndarray:
        from captum.attr import IntegratedGradients
        ig = ATTRIBUTION_CACHE.get(f"IntegratedGradients{self.model_id}", None)
        if ig is None:
            ig = IntegratedGradients(self.model)
            ATTRIBUTION_CACHE.set(f"IntegratedGradients{self.model_id}", ig)
        # attr: (H,W,C)
        attr = self.attribute_image_features(ig,
                                             input=tensor_image,
                                             target=label,
                                             baselines=tensor_image * 0,
                                             n_steps=self.options["n_steps"],
                                             method=self.options["method"])
        return attr


class Saliency(BaseAttribution):
    def __init__(self, req: dict):
        super().__init__(req)

    def attribute(self, tensor_image: torch.Tensor, label: int) -> np.ndarray:
        from captum.attr import Saliency
        ig = ATTRIBUTION_CACHE.get(f"Saliency{self.model_id}", None)
        if ig is None:
            ig = Saliency(self.model)
            ATTRIBUTION_CACHE.set(f"Saliency{self.model_id}", ig)
        # attr: (H,W,C)
        attr = self.attribute_image_features(ig,
                                             input=tensor_image,
                                             target=label)
        return attr


class DeepLift(BaseAttribution):
    def __init__(self, req: dict):
        super().__init__(req)

    def attribute(self, tensor_image: torch.Tensor, label: int) -> np.ndarray:
        from captum.attr import DeepLift
        ig = ATTRIBUTION_CACHE.get(f"DeepLift{self.model_id}", None)
        if ig is None:
            ig = DeepLift(self.model)
            ATTRIBUTION_CACHE.set(f"DeepLift{self.model_id}", ig)
        # attr: (H,W,C)
        attr = self.attribute_image_features(ig,
                                             input=tensor_image,
                                             target=label)
        return attr


class Occlusion(BaseAttribution):
    def __init__(self, req: dict):
        super().__init__(req)

    def attribute(self, tensor_image: torch.Tensor, label: int) -> np.ndarray:
        from captum.attr import Occlusion
        ig = ATTRIBUTION_CACHE.get(f"Occlusion{self.model_id}", None)
        if ig is None:
            ig = Occlusion(self.model)
            ATTRIBUTION_CACHE.set(f"Occlusion{self.model_id}", ig)
        # attr: (H,W,C)
        sliding_window_shapes = self.options["sliding_window_shapes"]
        strides = self.options["strides"]
        attr = self.attribute_image_features(ig,
                                             input=tensor_image,
                                             target=label,
                                             sliding_window_shapes=(3, sliding_window_shapes, sliding_window_shapes),
                                             strides=(3, strides, strides),
                                             show_progress=True
                                             )
        return attr
