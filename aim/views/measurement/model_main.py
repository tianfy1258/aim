import json

from celery.result import AsyncResult

from .model_task import *
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


def terminate_model_measurement_task(request):
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
        task_obj = ModelMeasurementTask.objects.get(task_id=task_id)
        task_obj.task_status = 'REVOKED'
        task_obj.save()
        return success_response({})
    elif task.status == "PROGRESS":
        task_interrupt(task_id)
        return success_response({})
    else:
        return error_response({}, error_message="任务已开始，无法取消")


def model_measurement_task_create(request):
    if request.method != 'POST':
        return error_response({})

    req = json.loads(request.body)
    dataset_id = req['dataset_id']
    model_id = req['model_id']
    deep_model = DeepModel.objects.get(model_id=model_id)
    dataset = Dataset.objects.get(dataset_id=dataset_id)
    # extract properties of DeepModel object
    deep_model_dict = {
        'model_id': deep_model.model_id,
        'model_name': deep_model.model_name,
        'model_description': deep_model.model_description,
        'model_dataset': deep_model.model_dataset,
        'is_predefine': deep_model.is_predefine,
        'is_use_function': deep_model.is_use_function,
        'model_size': deep_model.model_size,
        'model_type': deep_model.model_type,
        'model_output_shape': deep_model.model_output_shape,
        'model_filename': deep_model.model_filename,
        'model_classname': deep_model.model_classname,
        'model_processor': deep_model.model_processor
    }

    # extract properties of Dataset object
    dataset_dict = {
        'dataset_id': dataset.dataset_id,
        'dataset_name': dataset.dataset_name,
        'dataset_description': dataset.dataset_description,
        'dataset_filename': dataset.dataset_filename,
        'dataset_size': dataset.dataset_size,
        'dataset_instances': dataset.dataset_instances,
        'labels_num': dataset.labels_num,
        'dataset_type': dataset.dataset_type
    }
    req['dataset'] = dataset_dict
    req['deep_model'] = deep_model_dict
    req['create_user'] = request.session.get('user_id')
    task_id = model_measurement.apply_async(args=(req,), retry=False, countdown=5)

    # task_id = add.apply_async(args=(3, 4))
    LOGGER.info("task_id: %s" % task_id)

    return success_response({
        'task_id': task_id,
    })
