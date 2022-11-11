from backend.settings import MODEL_PATH
import torch
import importlib
from .logger import LOGGER
from typing import *
from .exception import CustomException
from .cache import DeepModelCache
MODEL_CACHE = DeepModelCache()

def parse_model(filename: str, classname: str, funcname: str) -> Tuple[torch.nn.Module,
                                                                       Callable,
                                                                       List]:
    """
    :param filename: 模型文件名
    :param classname: 模型类名
    :param funcname: 图像处理函数名
    :return:
    """
    # 此时假定文件符合要求，文件已经预先校验
    model_path = fr"{MODEL_PATH}\{filename}.pt"
    class_path = fr"{MODEL_PATH}\{filename}.txt"
    net = MODEL_CACHE.get(model_path,None)
    if net is None:
        try:
            Net = importlib.import_module(fr"deep_model.{filename}").__getattribute__(classname)
            LOGGER.debug(fr"热加载模块：deep_model.{filename} 模型对象：{classname}")
        except Exception as e:
            LOGGER.error(fr"模型定义文件加载失败！热加载模块：deep_model.{filename} 模型对象：{classname}", exc_info=True)
            raise CustomException("模型定义文件加载失败！")
        try:
            net: torch.nn.Module = Net()
            net.load_state_dict(torch.load(model_path))
            LOGGER.debug(fr"加载模型成功")
            MODEL_CACHE.set(model_path, net)
        except Exception as e:
            LOGGER.error(fr"模型文件加载失败！", exc_info=True)
            raise CustomException("模型文件加载失败！")
    else:
        LOGGER.debug(fr"使用缓存的模型")



    try:
        func = importlib.import_module(fr"deep_model.{filename}").__getattribute__(funcname)
        LOGGER.debug(fr"读取模型图片处理方法成功")
    except Exception as e:
        LOGGER.error(fr"模型图片处理方法读取失败！", exc_info=True)
        raise CustomException("模型图片处理方法读取失败！")

    try:
        with open(class_path) as f:
            classes = f.readlines()
        classes = [x for x in classes if x]
        LOGGER.debug(fr"读取模型类别成功")
    except Exception as e:
        LOGGER.error(fr"模型类别读取失败！", exc_info=True)
        raise CustomException("模型类别读取失败！")

    return net, func, classes
