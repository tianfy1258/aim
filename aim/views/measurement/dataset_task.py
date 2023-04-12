import json

from aim.utils import *
from aim.models import *
import pandas as pd
import numpy as np
import random as r
from celery import shared_task
from aim.utils.task import is_task_interrupt, CustomDatasetMeasurementTask
import piq


@shared_task(bind=True, base=CustomDatasetMeasurementTask)
def dataset_measurement(self, req):
    import time
    task_start_time = time.time()
    dataset = req['dataset']
    enable_random = req['enable_random']
    enable_compare = req['enable_compare']
    random_seed = req['random_seed']
    sample_count = req['sample_count']

    # 从缓存中获取zip数据集文件
    zfile = get_zipfile(dataset['dataset_filename'])
    # 根据采样数量获得要分析的图片samples
    filenames = [x.filename for x in zfile.filelist]
    if enable_random:
        r.seed(random_seed)
    samples = r.sample(filenames, k=sample_count)

    interrupted_progress = None

    dataset_li = []
    gradient_values = []
    brisque_values = []
    for i, (filename, image) in enumerate(zip(samples, extract_images(zfile, samples, use_cache=False))):
        # filename: str
        # image: PIL.Image
        # 获取numpy格式图片
        image_arr: np.ndarray = np.array(image)
        dataset_li.append(image_arr)

        image_L = np.array(image.convert('L'))
        from skimage.filters import sobel
        # 计算x和y方向的梯度
        grad_x = sobel(image_L, axis=1, mode='reflect')
        grad_y = sobel(image_L, axis=0, mode='reflect')

        # 计算梯度幅值
        grad_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

        # 计算平均梯度
        mean_gradient = np.mean(grad_magnitude)
        gradient_values.append(mean_gradient)
        # 计算BRISQUE index
        image_tensor = torch.from_numpy(image_L).float().unsqueeze(0).unsqueeze(0)
        # kernel_size是用于比较的滑动窗口的边长，设置为7；kernel_sigma是正态分布的参数，设置为7/6；data_range是图像的最大值范围，设置为255
        brisque_index: torch.Tensor = piq.brisque(image_tensor, 7, 7 / 6, 255, 'mean')
        brisque_values.append(brisque_index)

        if i % 100 == 0:
            interrupted_progress = dict(current=i, total=sample_count)
            self.update_state(state='PROGRESS', meta=interrupted_progress)
            if is_task_interrupt(self.request.id):
                break

    # stack dataset_li: (sample_count, height, width, channel)
    dataset_arr = np.stack(dataset_li)
    """
    假设 dataset_arr 是一个形状为 (N, H, W, C) 的 numpy 数组，其中：

    N 是样本数量；
    H 是每个样本的高度；
    W 是每个样本的宽度；
    C 是每个样本的通道数。
    np.mean(dataset_arr, axis=(0, 1, 2)) 
    表示对 dataset_arr 沿着前三个轴（即 N、H 和 W 轴）进行均值计算。
    具体来说，这将得到一个形状为 (C,) 的 numpy 数组，其中每个元素是每个通道在所有样本上的均值。
    换句话说，这个数组包含了所有像素值的平均值，其中不同通道的像素被视为不同的特征。

    例如，如果 dataset_arr 表示了一组 RGB 图像，
    那么 np.mean(dataset_arr, axis=(0, 1, 2)) 将得到一个形状为 (3,) 的数组，
    其中第一个元素是所有红色通道像素值的平均值，第二个元素是所有绿色通道像素值的平均值，第三个元素是所有蓝色通道像素值的平均值。
    
    """
    mean = np.mean(dataset_arr, axis=(0, 1, 2))
    std = np.std(dataset_arr, axis=(0, 1, 2))
    min = np.min(dataset_arr, axis=(0, 1, 2))
    max = np.max(dataset_arr, axis=(0, 1, 2))
    median = np.median(dataset_arr, axis=(0, 1, 2))
    gradient = {
        "mean": np.mean(gradient_values),
        "std": np.std(gradient_values),
        "min": np.min(gradient_values),
        "max": np.max(gradient_values),
    }
    brisque = {
        "mean": np.mean(brisque_values),
        "std": np.std(brisque_values),
        "min": np.min(brisque_values),
        "max": np.max(brisque_values),
    }
    task_end_time = time.time()
    status = "SUCCESS" if not is_task_interrupt(self.request.id) else "INTERRUPTED"
    res = {
            "result": {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": min.tolist(),
                "max": max.tolist(),
                "median": median.tolist(),
                "gradient": gradient,
                "brisque": brisque,
            },
            "task_start_time": task_start_time,
            "task_end_time": task_end_time,
            "status": status,
            "interrupted_progress": interrupted_progress,
        }

    # Define a custom encoder function for numpy arrays
    def numpy_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # convert numpy array to list
        elif isinstance(obj, np.float32):
            return float(obj)  # convert numpy float32 to float
        else:
            raise TypeError(f"{obj} is not JSON serializable")
    # 令人迷惑的操作，但此程序依赖于此操作
    return json.loads(
        json.dumps(
            res, default=numpy_encoder
        )
    )
