from celery.signals import before_task_publish, task_success, task_failure

from aim.utils import *
from aim.models import *
from celery.result import AsyncResult
import json
import time
import redis

# 连接到Redis服务器
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 定义一个键，用于存储中断任务列表
interrupted_tasks_key = 'interrupted_tasks'


def task_interrupt(task_id: str) -> None:
    """
    将给定的task_id加入到中断列表。

    参数：
        task_id (str): 要中断的任务ID。

    返回：
        无
    """
    # 将任务ID添加到Redis中断任务集合
    redis_client.sadd(interrupted_tasks_key, task_id)


def is_task_interrupt(task_id: str) -> bool:
    """
    检查给定的task_id是否已经被中断。

    参数：
        task_id (str): 要检查的任务ID。

    返回：
        bool: 如果任务已经被中断，返回True，否则返回False。
    """
    # 检查任务ID是否在Redis中断任务集合中
    return redis_client.sismember(interrupted_tasks_key, task_id)


def _get_task_status(task_id):
    # list
    if isinstance(task_id, list):
        result_li = []
        for _id in task_id:
            task = AsyncResult(_id)
            result_li.append(
                {
                    'task_status': task.status,
                    'task_result': task.result,
                    'task_id': task_id,
                    'task_name': task.name,
                    'task_date_done': task.date_done,
                }
            )
        return result_li
    # str
    task = AsyncResult(task_id)
    return {
        'task_status': task.status,
        'task_result': task.result,
        'task_id': task_id,
        'task_name': task.name,
        'task_date_done': task.date_done,
    }


@before_task_publish.connect
def before_task(sender=None, headers=None, body=None, **kwargs):
    task_id = headers['id']
    argsrepr = eval(headers['argsrepr'])[0]
    if sender == 'aim.views.measurement.model_task.model_measurement':
        try:
            task = ModelMeasurementTask(
                task_id=task_id,
                task_name=argsrepr['task_name'],
                task_description=argsrepr['task_description'],
                measure_method=argsrepr['method_list'],
                dataset_id_id=int(argsrepr['dataset_id']),
                model_id_id=int(argsrepr['model_id']),
                enable_random=argsrepr['enable_random'],
                sample_count=argsrepr['sample_count'],
                random_seed=argsrepr['random_seed'],
                task_status='PENDING',
                task_result='',
                task_traceback='',
                create_user_id=argsrepr['create_user'],
                update_user_id=argsrepr['create_user'],
            )
            task.save()
            LOGGER.debug(f"任务信息已写入db: {task_id} {argsrepr['task_name']} ")
        except Exception as e:
            import traceback
            LOGGER.error(e)
            LOGGER.error(traceback.format_exc())
    elif sender == 'aim.views.measurement.dataset_task.dataset_measurement':
        try:
            task = DatasetMeasurementTask(
                task_id=task_id,
                task_name=argsrepr['task_name'],
                task_description=argsrepr['task_description'],
                single_measure_method=argsrepr['single_method_list'],
                dataset_id_id=int(argsrepr['dataset_id_1']),
                enable_compare=argsrepr['enable_compare'],
                enable_random=argsrepr['enable_random'],
                # dataset_compare_id_id= [待开发]
                sample_count=argsrepr['sample_count'],
                random_seed=argsrepr['random_seed'],
                task_status='PENDING',
                task_result='',
                task_traceback='',
                create_user_id=argsrepr['create_user'],
                update_user_id=argsrepr['create_user'],
            )
            task.save()
            LOGGER.debug(f"任务信息已写入db: {task_id} {argsrepr['task_name']} ")
        except Exception as e:
            import traceback
            LOGGER.error(e)
            LOGGER.error(traceback.format_exc())


from celery import Task


class CustomModelMeasurementTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        task = ModelMeasurementTask.objects.filter(task_id=task_id)
        if len(task) == 1:
            task = task[0]
            info = _get_task_status(task.task_id)
            result = info['task_result']
            if result['status'] == 'INTERRUPTED':
                task.task_status = 'INTERRUPTED'
            else:
                task.task_status = 'SUCCESS'
            task.task_result = json.dumps(result)
            LOGGER.error(task)
            task.save()

        return super(CustomModelMeasurementTask, self).on_success(retval, task_id, args, kwargs)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        task = ModelMeasurementTask.objects.filter(task_id=task_id)
        if len(task) == 1:
            task = task[0]
            task.task_status = 'FAILURE'
            task.task_traceback = einfo
            task.save()
        return super(CustomModelMeasurementTask, self).on_failure(exc, task_id, args, kwargs, einfo)


class CustomDatasetMeasurementTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        task = DatasetMeasurementTask.objects.filter(task_id=task_id)
        if len(task) == 1:
            task = task[0]
            info = _get_task_status(task.task_id)
            result = info['task_result']
            if result['status'] == 'INTERRUPTED':
                task.task_status = 'INTERRUPTED'
            else:
                task.task_status = 'SUCCESS'
            task.task_result = json.dumps(result)
            LOGGER.error(task)
            task.save()

        return super(CustomDatasetMeasurementTask, self).on_success(retval, task_id, args, kwargs)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        task = DatasetMeasurementTask.objects.filter(task_id=task_id)
        if len(task) == 1:
            task = task[0]
            task.task_status = 'FAILURE'
            task.task_traceback = einfo
            task.save()
        return super(CustomDatasetMeasurementTask, self).on_failure(exc, task_id, args, kwargs, einfo)
