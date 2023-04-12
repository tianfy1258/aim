from aim.utils import *
from aim.models import *
import pandas as pd
import numpy as np
import random as r
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score, accuracy_score
# , roc_auc_score, roc_curve, multilabel_confusion_matrix, \
#     top_k_accuracy_score
from celery import shared_task

from aim.utils.task import is_task_interrupt, CustomModelMeasurementTask


@shared_task(bind=True, base=CustomModelMeasurementTask)
def model_measurement(self, req):
    import time
    task_start_time = time.time()
    method_list = req['method_list']
    task_name = req['task_name']
    task_description = req['task_description']
    enable_random = req['enable_random']
    random_seed = req['random_seed']
    sample_rate = req['sample_rate']
    sample_count = req['sample_count']

    deep_model = req['deep_model']
    # 获取数据集信息
    dataset = req['dataset']
    # 加载数据集标签信息
    dataset_df = pd.read_csv(fr"{DATA_PATH}\{dataset['dataset_filename']}.csv")
    dataset_df = dataset_df.set_index("filename")
    # 加载模型，net, func, classes
    model, func, classes = parse_model(deep_model['model_filename'],
                                       deep_model['model_classname'],
                                       deep_model['model_processor'],
                                       is_use_function=deep_model['is_use_function']
                                       )
    # 从缓存中获取zip数据集文件
    zfile = get_zipfile(dataset['dataset_filename'])
    # 根据采样数量获得要分析的图片samples
    filenames = [x.filename for x in zfile.filelist]
    if enable_random:
        r.seed(random_seed)
    samples = r.sample(filenames, k=sample_count)

    y_true = []
    y_pred = []
    # 准备数据集
    interrupted_progress = None
    for i, (filename, image) in enumerate(zip(samples, extract_images(zfile, samples, use_cache=False))):
        # filename: str
        # image: PIL.Image
        # 获取numpy格式图片
        image_arr: np.ndarray = np.array(image)
        # 使用模型的processor对tensor进行处理，得到tensor格式的处理后图片
        tensor_image: torch.Tensor = func(image_arr)
        # 如果不是4维，多加一个Batch维度
        if len(tensor_image.shape) == 3:
            # tensor_image :(1,C,H,W)
            tensor_image = tensor_image.unsqueeze(0)
        # 获得模型对该图片的分类结果（int值）
        # (1,C,H,W) -> (1,output_dim)
        with torch.no_grad():
            _outputs: torch.Tensor = model(tensor_image)
        if not isinstance(_outputs, torch.Tensor):
            raise CustomException(f"模型输出为{type(_outputs)}，应该为torch.Tensor")
        if not (_outputs.shape[0] == 1 and len(_outputs.shape) == 2):
            raise CustomException(f"模型输出为大小为{tuple(_outputs.shape)}，应该为(1,标签个数)")
        # (1,output_dim) -> int
        _, output = torch.max(_outputs, 1)
        index = output.item()
        score = torch.softmax(_outputs, 1)[0, index].item()
        # 获得模型对该图片的分类
        predict_label = classes[index]
        # 获得图片本身的分类
        label = dataset_df.loc[filename][0]
        # 计算归因结果并写入缓存
        y_pred.append(index)
        y_true.append(classes.index(label))
        # import time
        # time.sleep(0.5) # for debug
        if i % 100 == 0:
            interrupted_progress = dict(current=i, total=sample_count)
            self.update_state(state='PROGRESS', meta=interrupted_progress)
            if is_task_interrupt(self.request.id):
                break

    def find_k_accuracy(y_true, y_pred, classes, k, which="lowest"):
        # 初始化字典
        class_stats = {i: {'correct': 0, 'total': 0} for i in range(len(classes))}

        # 统计每个类的正确预测次数和总预测次数
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                class_stats[true]['correct'] += 1
            class_stats[true]['total'] += 1

        # 计算正确率
        class_accuracy = [
            # ["dog",5,10,0.5]
            [classes[i], class_stats[i]['correct'], class_stats[i]['total'],
             class_stats[i]['correct'] / class_stats[i]['total']]
            if class_stats[i]['total'] != 0
            else
            [classes[i], class_stats[i]['correct'], class_stats[i]['total'], 0]
            for i in class_stats
        ]
        class_accuracy = [x for x in class_accuracy if x[2] != 0]
        # 对正确率进行排序
        lowest_k_accuracy = sorted(class_accuracy, key=lambda x: (x[3], -x[2]))
        highest_k_accuracy = sorted(class_accuracy, key=lambda x: (-x[3], -x[2]))
        if which == "lowest":
            # 输出前k个正确率最低的标签
            return lowest_k_accuracy[:k]
        if which == "highest":
            return highest_k_accuracy[:k]

    k = 30
    lowest_k_accuracy = find_k_accuracy(y_true, y_pred, classes, k, which="lowest")
    highest_k_accuracy = find_k_accuracy(y_true, y_pred, classes, k, which="highest")

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    kappa = cohen_kappa_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    task_end_time = time.time()

    status = "SUCCESS" if not is_task_interrupt(self.request.id) else "INTERRUPTED"

    return {
        "tableData": [
            {"label": "宏平均精度", "value": f"{precision_macro:.4f}"},
            {"label": "宏平均召回率", "value": f"{recall_macro:.4f}"},
            {"label": "宏平均F1得分", "value": f"{f1_macro:.4f}"},
            {"label": "微平均精度", "value": f"{precision_micro:.4f}"},
            {"label": "微平均召回率", "value": f"{recall_micro:.4f}"},
            {"label": "微平均F1得分", "value": f"{f1_micro:.4f}"},
            {"label": "Cohen's Kappa系数", "value": f"{kappa:.4f}"},
            {"label": "准确率", "value": f"{accuracy:.4f}"},
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
