import os

from django.db import transaction

from aim.utils import *
from aim.models import *
from backend.settings import UPLOAD_PATH, DATA_PATH
import zipfile
import pandas as pd
import time
import inspect
import torch
import numpy as np


def after_upload_model(request):
    if request.method != 'POST':
        return error_response({})

    req = json.loads(request.body)
    upload_token = req["token"]
    pt_name = req['ptName']
    py_name = req['pyName']
    txt_name = req['txtName']
    pt_path = fr"{UPLOAD_PATH}\{upload_token}_{pt_name}"
    py_path = fr"{UPLOAD_PATH}\{upload_token}_{py_name}"
    txt_path = fr"{UPLOAD_PATH}\{upload_token}_{txt_name}"

    def process(request) -> Union[str, Tuple[list, list, int]]:

        req = json.loads(request.body)
        upload_token = req["token"]
        pt_name = req['ptName']
        py_name = req['pyName']
        txt_name = req['txtName']
        LOGGER.debug(f"ptName: {pt_name} txtName: {txt_name}  pyName:{py_name}")
        LOGGER.debug("尝试读取py文件")
        try:
            module = importlib.import_module(fr"upload.{upload_token}_{py_name[:-3]}")
        except Exception as e:
            LOGGER.error(exc_info=True, msg=f"读取python文件失败: {e}")
            return "python文件读取失败！"

        try:
            functions = inspect.getmembers(module, predicate=inspect.isfunction)
            models = inspect.getmembers(module, predicate=inspect.isclass)

        except Exception as e:
            LOGGER.error(exc_info=True, msg=f"解析Python文件失败: {e}")
            return "python文件解析失败！"

        LOGGER.debug("尝试读取txt文件")
        try:
            lines = None
            with open(fr"{UPLOAD_PATH}\{upload_token}_{txt_name}") as f:
                lines = f.readlines()
            lines = [x.strip() for x in lines if x]
        except Exception as e:
            LOGGER.error(exc_info=True, msg=f"读取txt文件失败: {e}")
            return "txt文件读取失败！"

        return [x[0] for x in models], [x[0] for x in functions], len(lines)

    r = process(request)

    if isinstance(r, str):
        for filepath in [pt_path, py_path, txt_path]:
            if os.path.exists(filepath):
                os.remove(filepath)
                LOGGER.info(f"删除 {filepath}")
        return error_response({}, r)

    models, functions, labels_num = r
    res = {
        "upload_token": upload_token,
        "functions": functions,
        "labels_num": labels_num,
        "models": models,
        "pt_path":pt_path,
        "py_path":py_path,
        "py_name":py_name,
        "txt_path":txt_path,
    }

    return success_response(res)


def check_is_py_valid(request):
    if request.method != 'POST':
        return error_response({})

    req = json.loads(request.body)
    upload_token = req["upload_token"]
    py_name = req['py_name']
    image_processor = req['image_processor']

    module = importlib.import_module(fr"upload.{upload_token}_{py_name[:-3]}")
    functions = inspect.getmembers(module, predicate=inspect.isfunction)
    try:
        function_index = [x[0] for x in functions].index(image_processor)
    except ValueError as e:
        LOGGER.error(exc_info=True, msg=f"未在{py_name}中找到函数{image_processor}")
        return error_response({}, f"未在{py_name}中找到函数{image_processor}")

    func = functions[function_index][1]

    test_numpy = np.random.random((114, 514, (514 - 114) % 397))  # (H, W, C) 测试矩阵
    try:
        output: torch.Tensor = func(test_numpy)
    except Exception as e:
        LOGGER.error(exc_info=True, msg=f"处理方法调用失败:{e}")
        return error_response({}, f"处理方法调用失败，{e}")

    if not isinstance(output, torch.Tensor):
        LOGGER.error(f"检测到处理方法输出类型为{type(output)}，应该为torch.Tensor")
        return error_response({}, f"检测到处理方法输出类型为{type(output)}，应该为torch.Tensor")

    if len(output.shape) != 3:
        LOGGER.error(f"检测到处理方法输出维度为{len(output.shape)}，应该为3维")
        return error_response({}, f"检测到处理方法输出维度为{len(output.shape)}，应该为3维")

    if output.shape[0] != 3:
        LOGGER.error(f"检测到处理方法输出的结果不是(C, H, W)形式，请保证通道数量在第一维")
        return error_response({}, f"检测到处理方法输出的结果不是(C, H, W)形式，请保证通道数量在第一维")

    return success_response({})


@transaction.atomic
def create_model(request):
    if request.method != 'POST':
        return error_response({})

    req = json.loads(request.body)
    try:
        model_classname= req["model_classname"]
        image_processor= req["image_processor"]
        model_name= req["model_name"]
        model_description= req["model_description"]
        upload_token= req["upload_token"]
        labels_num= req["labels_num"]
        py_name= req["py_name"]
        pt_path= req["pt_path"]
        py_path= req["py_path"]
        txt_path= req["txt_path"]
        create_user = User.objects.get(user_id=request.session.get('user_id'))
    except Exception as e:
        LOGGER.error(exc_info=True, msg=f"参数错误: {e}")
        return error_response({}, "系统错误，解析参数时出现问题")

    model_filename = md5(fr"{create_user.username}_{time.time()}_{model_name}")
    model_filepath = fr"{MODEL_PATH}\{model_filename}"
    try:
        r = DeepModel.objects.filter(model_name=model_name)
        if len(r) > 0:
            LOGGER.debug("重复的模型名称")
            return error_response({}, "模型名称已存在")

        m = DeepModel(
            model_name=model_name,
            model_description=model_description,
            model_size=os.path.getsize(pt_path),
            model_type="自定义模型",
            model_output_shape=labels_num,
            model_filename=model_filename,
            model_classname=model_classname,
            model_processor=image_processor,
            create_user=create_user,
            update_user=create_user
        )
        m.save()
    except Exception as e:
        LOGGER.error(exc_info=True, msg=f"创建模型失败，无法写入模型对象 {e}")
        return error_response({}, "系统错误，无法写入模型对象")

    try:
        import shutil
        shutil.move(pt_path, fr"{model_filepath}.pt")
        shutil.move(py_path, fr"{model_filepath}.py")
        shutil.move(txt_path, fr"{model_filepath}.txt")
    except Exception as e:
        LOGGER.error(exc_info=True, msg="创建模型失败，无法获取到上传文件")
        return error_response({}, "系统错误，无法获取到上传文件")

    return success_response({})
