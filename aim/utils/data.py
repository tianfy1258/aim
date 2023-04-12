import json

from django.http import HttpRequest
from django.db.models.manager import Manager
from .request import *
from aim.models import *
from .logger import LOGGER
import os
from backend.settings import DATA_PATH,MODEL_PATH

class DataQuery:
    """
    model: django.db.models
    projection: list 要选择的列，默认为None（表示全选）
    additional_projection: list 要选择的外键列，默认选择create_user__username, update_user__username
    handler: function(request) -> None 查询前的钩子函数
    """

    @staticmethod
    def query_builder(model, projection: list = None, use_additional_projection:bool = True,additional_projection: list = None, handler=None,after_data_return=None):
        if not additional_projection:
            additional_projection = ["create_user__username", "update_user__username"]
        if not projection:
            projection = [f.attname for f in model._meta.concrete_fields]
        if additional_projection and use_additional_projection:
            projection += additional_projection
        return DataQuery(model.objects, projection, handler,after_data_return).query

    def __init__(self, objects: Manager, projection, handler=None,after_data_return=None):
        self.objects = objects
        self.pageSize = 100
        self.pageNum = 1
        self.filter = {}
        self.order_by = None
        self.handler = handler
        self.projection = projection
        self.request = None
        self.after_data_return = after_data_return

    def set_request(self, request: HttpRequest, ):
        if request.method == 'GET':
            self.pageSize = int(request.GET.get(key="pageSize", default=100))
            self.pageNum = int(request.GET.get(key="pageNum", default=1))
            self.filter = request.GET.get(key="filter", default={})
            self.order_by = request.GET.get(key="orderBy", default=None)
        elif request.method == 'POST':
            req = json.loads(request.body)
            self.pageSize = int(req.get("pageSize", 100))
            self.pageNum = int(req.get("pageNum", 1))
            self.filter = req.get("filter", {})
            self.order_by = req.get("orderBy", None)
            LOGGER.debug(f"query body: {req}")
        self.request = request

        if self.handler is not None:
            self.handler(request)

    def all(self):

        if self.order_by:
            self.objects = self.objects.order_by(self.order_by)
        return \
            self.objects.filter(**self.filter).values(*self.projection) \
                [self.pageSize * (self.pageNum - 1):self.pageSize * self.pageNum], \
            self.objects.filter(**self.filter).values(*self.projection).count()

    def query(self, request):
        self.set_request(request)

        li, count = self.all()
        res = {
            "data": list(li),
            "total": count
        }
        if self.after_data_return:
            self.after_data_return(res)

        return success_response(res)


class DataDelete:
    """
    before_delete: function(request) -> None 删除数据之前调用的钩子函数
    """

    @staticmethod
    def query_builder(model):
        if model == Dataset:
            def delete_files(request):
                req = json.loads(request.body)
                pk = model._meta.concrete_fields[0].attname
                obj = model.objects.get(**{pk: req["id"]})
                filepath = fr"{DATA_PATH}\{obj.dataset_filename}"
                for type_ in ["zip", "csv"]:
                    if os.path.exists(fr"{filepath}.{type_}"):
                        os.remove(fr"{filepath}.{type_}")
                        LOGGER.info(fr"删除 {filepath}.{type_}")

            return DataDelete(model, before_delete=delete_files).query
        elif model == DeepModel:
            def delete_files(request):
                req = json.loads(request.body)
                pk = model._meta.concrete_fields[0].attname
                obj = model.objects.get(**{pk: req["id"]})
                filepath = fr"{MODEL_PATH}\{obj.model_filename}"
                for type_ in ["py", "pt","txt"]:
                    if os.path.exists(fr"{filepath}.{type_}"):
                        os.remove(fr"{filepath}.{type_}")
                        LOGGER.info(fr"删除 {filepath}.{type_}")

            return DataDelete(model, before_delete=delete_files).query
        else:
            return DataDelete(model).query

    def __init__(self, model: models.Model, before_delete=None):
        self.model = model
        self.before_delete = before_delete

    def query(self, request):
        if request.method != "POST":
            return error_response({})

        try:
            req = json.loads(request.body)
            pk = self.model._meta.concrete_fields[0].attname
            obj = self.model.objects.get(**{pk:req["id"]})
            if self.before_delete:
                self.before_delete(request)
            obj.delete()
        except Exception as e:
            LOGGER.error(exc_info=True,msg=f"删除失败: {e}")
            return error_response({}, "删除失败")
        return success_response({})


class DataUpdate:

    @staticmethod
    def query_builder(model):
        return DataUpdate(model.objects).query

    def __init__(self, objects: Manager):
        self.objects = objects

    def query(self, request):
        if request.method != "POST":
            return error_response({})
        try:
            req = json.loads(request.body)
            obj = self.objects.get(req["data"])
            obj.update(**req["data"])
        except Exception as e:
            LOGGER.error(exc_info=True,msg=f"修改失败: {e}")
            return error_response({}, "更新失败")

        return success_response({})
