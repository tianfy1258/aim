import json
from typing import *

from django.http import HttpResponse

SUCCESS_CODE = 200
ERROR_CODE = 400
INVALID_CODE = 401
RETRY_CODE = 402


def success_response(response: dict):
    response['code'] = SUCCESS_CODE
    return HttpResponse(json.dumps(response, default=str))


def error_response(response: dict, error_message: str = "", duration: int = 3000):
    response['code'] = ERROR_CODE
    response['error_message'] = error_message
    response['duration'] = duration
    return HttpResponse(json.dumps(response, default=str))


# 目前任务都只是在内存中
# 等用异步任务队列，也没这玩意啥事了。自旋锁，是屎山代码第一步，所以我勇敢的踏了出来！
#  任务运行周期： ------ ======== ------
#  获取状态：    ---↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑----
#  当任务还没运行或运行结束时获取状态，就是retry，最多5次
#  不要问为什么不能任务开启时，再告诉前端获取状态，别问，问就是没时间写。
def retry_response(response: dict):
    response['code'] = RETRY_CODE
    return HttpResponse(json.dumps(response, default=str))


def invalid_response(response: dict, error_message: str = "登录信息已失效，请重新登录"):
    response['code'] = INVALID_CODE
    response['error_message'] = error_message
    return HttpResponse(json.dumps(response, default=str))
