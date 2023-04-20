"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.shortcuts import render

from django.views.generic import TemplateView
from django.urls import path, include, re_path
from aim.views import *
from aim.utils import DataQuery
import inspect
from aim import models
from backend.settings import DEBUG, SILK

# 所有Models的查询接口
database_query = [
    path(f'get{classname}', DataQuery.query_builder(
        class_
    ))
    for classname, class_ in inspect.getmembers(models, inspect.isclass)
]
# 所有Models的删除接口
database_delete = [
    path(f'delete{classname}', DataDelete.query_builder(
        class_
    ))
    for classname, class_ in inspect.getmembers(models, inspect.isclass)
]
# 所有Models的修改接口
database_update = [
    path(f'update{classname}', DataUpdate.query_builder(
        class_
    ))
    for classname, class_ in inspect.getmembers(models, inspect.isclass)
]

urlpatterns = [
    # 登录相关
    path('login', login),
    path('logout', logout),
    path('validToken', valid_token),

    # 数据库查询相关, 名称为 get{ModelName}, 例如getDataset
    *database_query,
    # 数据库删除相关, 名称为 delete{ModelName}, 例如deleteDataset
    *database_delete,
    # 数据库修改相关, 名称为 update{ModelName}, 例如updateDataset
    *database_update,
    # 数据库下载
    path("datasetDownload", dataset_download),

    # 文件上传相关
    path('upload', upload),
    path('afterUploadDataset', after_upload_dataset),
    path('afterUploadModel', after_upload_model),
    path('checkIsPyValid', check_is_py_valid),
    path('createDataset', create_dataset),
    path('createModel', create_model),
    # 可解释分析相关
    path('attribute', attribute),
    path('getImage', get_image),
    # 覆盖测试相关
    path('coverage', coverage),
    path('getStatus', get_status),
    path('terminateCoverageTask', terminate_coverage_task),
    # 可视化相关
    path('getFile/<str:fn>', get_file),
    path('getJsonImage', get_json_imagenet_1k),
    path('empty', lambda x: success_response({})),
    # 度量相关
    path('createModelMeasurementTask', model_measurement_task_create),
    path('createDatasetMeasurementTask', dataset_measurement_task_create),
    # 任务相关
    path('getModelMeasurementTaskList', DataQuery.query_builder(
        ModelMeasurementTask,
        after_data_return=query_redis_task,
        use_additional_projection=True,
        additional_projection=['dataset_id__dataset_name', 'model_id__model_name',"create_user__username", "update_user__username"]
    )),
    path('getDatasetMeasurementTaskList', DataQuery.query_builder(
        DatasetMeasurementTask,
        after_data_return=query_redis_task,
        use_additional_projection=True,
        additional_projection=['dataset_id__dataset_name', "create_user__username",
                               "update_user__username"]
    )),
    path('terminateModelMeasurementTask', terminate_model_measurement_task),
    path('terminateDatasetMeasurementTask', terminate_dataset_measurement_task),
    # 选项获取相关
    path('getDatasetOptions',
         DataQuery.query_builder(Dataset, ["dataset_id", "dataset_name", "dataset_instances","hashcode"],
                                 use_additional_projection=False)),
    path('getModelOptions',
         DataQuery.query_builder(DeepModel, ["model_id", "model_name"], use_additional_projection=False)),
    # 测试相关
    path('getTestImage/<str:fn>', get_test_image_file),
]
