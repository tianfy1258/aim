import json
from typing import *

from django.http import HttpResponse

SUCCESS_CODE = 200
ERROR_CODE = 400
INVALID_CODE = 401


def success_response(response: dict):
    response['code'] = SUCCESS_CODE
    return HttpResponse(json.dumps(response, default=str))


def error_response(response: dict, error_message: str = ""):
    response['code'] = ERROR_CODE
    response['error_message'] = error_message
    return HttpResponse(json.dumps(response, default=str))


def invalid_response(response: dict, error_message: str = "登录信息已失效，请重新登录"):
    response['code'] = INVALID_CODE
    response['error_message'] = error_message
    return HttpResponse(json.dumps(response, default=str))
