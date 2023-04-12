import json

from celery.result import AsyncResult

from .dataset_task import *
from aim.utils.task import _get_task_status, task_interrupt


def query_redis_task(res):
    tasks = res['data']
    for task in tasks:
        info = _get_task_status(task['task_id'])
        if task['task_status'] == 'PENDING':
            task['task_status'] = info['task_status']
            task['progress'] = info['task_result']
        if task['task_status'] == 'INTERRUPTED':
            result = json.loads(task['task_result'])
            task['progress'] = result['interrupted_progress']


def terminate_dataset_measurement_task(request):
    """
    终止任务, 仅限于未开始的任务
    若需要终止已开始的任务, 需要修改为 task.revoke(terminate=True)
    """
    if request.method != 'GET':
        return error_response({})
    task_id = request.GET.get("task_id")
    task = AsyncResult(task_id)

    if task.status == "PENDING":
        LOGGER.error("pending")
        task.revoke()
        task_obj = DatasetMeasurementTask.objects.get(task_id=task_id)
        task_obj.task_status = 'REVOKED'
        task_obj.save()
        return success_response({})
    elif task.status == "PROGRESS":
        task_interrupt(task_id)
        return success_response({})
    else:
        return error_response({}, error_message="任务已开始，无法取消")


def dataset_measurement_task_create(request):
    if request.method != 'POST':
        return error_response({})

    req = json.loads(request.body)
    dataset_id_1 = req['dataset_id_1']
    dataset_id_2 = req['dataset_id_2']
    dataset = Dataset.objects.get(dataset_id=dataset_id_1)
    # dataset_compare = Dataset.objects.get(dataset_id=dataset_id_2)

    dataset_dict = {
        'dataset_id': dataset.dataset_id,
        'dataset_name': dataset.dataset_name,
        'dataset_description': dataset.dataset_description,
        'dataset_filename': dataset.dataset_filename,
        'dataset_size': dataset.dataset_size,
        'dataset_instances': dataset.dataset_instances,
        'labels_num': dataset.labels_num,
        'hashcode': dataset.hashcode,
    }
    # dataset_compare_dict = {
    #     'dataset_id': dataset_compare.dataset_id,
    #     'dataset_name': dataset_compare.dataset_name,
    #     'dataset_description': dataset_compare.dataset_description,
    #     'dataset_filename': dataset_compare.dataset_filename,
    #     'dataset_size': dataset_compare.dataset_size,
    #     'dataset_instances': dataset_compare.dataset_instances,
    #     'labels_num': dataset_compare.labels_num,
    #     'hashcode': dataset_compare.hashcode,
    # }
    req['dataset'] = dataset_dict
    # req['dataset_compare'] = dataset_compare_dict
    req['create_user'] = request.session.get('user_id')
    task_id = dataset_measurement.apply_async(args=(req,), retry=False, countdown=5)

    # task_id = add.apply_async(args=(3, 4))
    LOGGER.info("task_id: %s" % task_id)
    return success_response({
        'task_id': task_id,
    })
