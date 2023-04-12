from aim.utils import *
from aim.models import *
import pandas as pd
import numpy as np
import random as r
from celery import shared_task
import cv2
from aim.utils.task import is_task_interrupt
from iqa.alg import snr, psnr, fsim, mse


@shared_task(bind=True)
def dataset_measurement(self, req):
    import time
    task_start_time = time.time()
    dataset = req['dataset']
    dataset_compare = req['dataset_compare']
    compare_method_list = req['compare_method_list']
    enable_random = req['enable_random']
    enable_compare = req['enable_compare']
    random_seed = req['random_seed']
    sample_count = req['sample_count']

    # 从缓存中获取zip数据集文件
    zfile = get_zipfile(dataset['dataset_filename'])
    # 从缓存中获取zip数据集文件
    zfile_compare = get_zipfile(dataset['dataset_filename']) if enable_compare else None
    # 根据采样数量获得要分析的图片samples
    filenames = [x.filename for x in zfile.filelist]
    if enable_random:
        r.seed(random_seed)
    samples = r.sample(filenames, k=sample_count)

    dataset_li = []
    dataset_compare_li = []
    mse_li = []
    psnr_li = []
    snr_li = []
    fsim_li = []
    res = {"compare_result": {}}
    for i, (filename, image) in enumerate(zip(samples, extract_images(zfile, samples, use_cache=False))):
        # filename: str
        # image: PIL.Image
        # 获取numpy格式图片
        image_arr: np.ndarray = np.array(image)
        dataset_li.append(image_arr)
        if enable_compare:
            image_compare = extract_one_image(zfile, filename, use_cache=False)
            image_compare_arr: np.ndarray = np.array(image_compare)
            dataset_compare_li.append(image_compare_arr)

            # 计算x和y方向的梯度
            grad_x = cv2.Sobel(image_compare, cv2.CV_16S, 1, 0)
            grad_y = cv2.Sobel(image_array, cv2.CV_64S, 0, 1)

            # 计算梯度幅值
            grad_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

            # 计算平均梯度
            mean_gradient = np.mean(grad_magnitude)
            mean_gradient_values.append(mean_gradient)


            if "mse" in compare_method_list:
                mse_value = mse(image_arr, image_compare_arr)
                mse_li.append(mse_value)
                res["compare_result"][filename] = {"mse": mse_value}
            if "psnr" in compare_method_list:
                psnr_value = psnr(image_arr, image_compare_arr)
                psnr_li.append(psnr_value)
                res["compare_result"][filename] = {"psnr": psnr_value}
            if "snr" in compare_method_list:
                snr_value = snr(image_arr, image_compare_arr)
                snr_li.append(snr_value)
                res["compare_result"][filename] = {"snr": snr_value}
            if "fsim" in compare_method_list:
                fsim_value = fsim(image_arr, image_compare_arr)
                fsim_li.append(fsim_value)
                res["compare_result"][filename] = {"fsim": fsim_value}

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

    task_end_time = time.time()
    status = "SUCCESS" if not is_task_interrupt(self.request.id) else "INTERRUPTED"

    return {
        "single": [
            {"label": "宏平均精度", "value": f"{precision_macro:.2f}"},
            {"label": "宏平均召回率", "value": f"{recall_macro:.2f}"},
            {"label": "宏平均F1得分", "value": f"{f1_macro:.2f}"},
            {"label": "微平均精度", "value": f"{precision_micro:.2f}"},
            {"label": "微平均召回率", "value": f"{recall_micro:.2f}"},
            {"label": "微平均F1得分", "value": f"{f1_micro:.2f}"},
            {"label": "Cohen's Kappa系数", "value": f"{kappa:.2f}"},
            {"label": "准确率", "value": f"{accuracy:.2f}"},
        ],
        "lowestChartData": {
            "yAxis": [x[0] for x in lowest_k_accuracy],
            "dataCorrect": [x[1] for x in lowest_k_accuracy],
            "dataTotal": [x[2] for x in lowest_k_accuracy],
            "dataAccuracy": [x[3] for x in lowest_k_accuracy],
        },
        "highestChartData": {
            "yAxis": [x[0] for x in highest_k_accuracy],
            "dataCorrect": [x[1] for x in highest_k_accuracy],
            "dataTotal": [x[2] for x in highest_k_accuracy],
            "dataAccuracy": [x[3] for x in highest_k_accuracy],
        },
        "task_start_time": task_start_time,
        "task_end_time": task_end_time,
        "status": status,
        "interrupted_progress": interrupted_progress,
    }
