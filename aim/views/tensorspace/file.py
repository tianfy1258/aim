from aim.models import *
from aim.utils import *
from django.http import FileResponse
from backend.settings import TENSOR_SPACE_PATH
import random as r
import numpy as np
import pandas as pd
from torchvision import transforms


def get_file(request, fn):
    filename = fn

    filepath = fr"{TENSOR_SPACE_PATH}\{filename}"
    try:
        def file_iter(fp, chunk_size=1024 * 1024 * 8):
            with open(fp, 'rb') as f:
                while True:
                    c = f.read(chunk_size)
                    if c:
                        yield c
                    else:
                        break
    except Exception as e:
        LOGGER.error(exc_info=True, msg=f"文件读取失败: {e}")
        return error_response({}, "文件获取失败")

    response = FileResponse(file_iter(filepath), filename=filename)
    response['Content-Disposition'] = f'attachment; filename={filename}'
    return response


LOGGER.warn("[tensorspace] 使用写死的数据集id。因为模块默认使用ImageNet_1k数据集，该数据在数据库中的id为5578，请保证存在，否则会报错")
# 加载数据集标签信息
dataset = Dataset.objects.get(dataset_id=5578)
dataset_df = pd.read_csv(fr"{DATA_PATH}\{dataset.dataset_filename}.csv")
dataset_df = dataset_df.set_index("filename")

def _scale(input, rmax=1, rmin=0):
    input_std = (input - np.min(input)) / (input.max() - input.min())
    input_scaled = input_std * (rmax - rmin) + rmin
    return input_scaled

def get_json_imagenet_1k(request):
    """
    随机返回一张json格式的图片
    (PIL.Image -> np.array -> np.array(flatten) -> json)
    """
    width = int(request.GET.get("width", 224))
    height = int(request.GET.get("height", 224))
    scale = request.GET.get("scale","false")
    scale = True if scale == "true" else False

    zfile = get_zipfile(dataset.dataset_filename)
    fileinfo = r.choice(zfile.filelist)
    # fileinfo = zfile.filelist[r.choice([6801])] # debug
    filename = fileinfo.filename
    LOGGER.debug(f"随机选择图片:{filename}")
    image = extract_one_image(zfile, filename, False)
    image_arr: np.ndarray = np.array(image.resize((width, height)))
    if scale:
        image_arr = _scale(image_arr)
    image_arr = image_arr.ravel()
    return success_response({
        "data": image_arr.tolist(),
        "label": dataset_df.loc[filename][0]
    })
