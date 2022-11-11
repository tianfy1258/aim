from aim.utils import *
from aim.models import *
from .main import *


def attribute(request):
    if request.method != 'POST':
        return error_response({})
    req = json.loads(request.body)
    attribution_method = req['attribution']
    attribution: BaseAttribution = None
    try:
        if attribution_method == 'Integrated Gradients':
            attribution = IntegratedGradients(req)
    except CustomException as e:
        LOGGER.error("自定义错误", exc_info=True)
        return error_response({}, str(e))
    except Exception as e:
        LOGGER.error("模型初始化失败", exc_info=True)
        return error_response({}, "模型初始化失败！")

    res = {
        "data": attribution.attribute_from_dataset()
    }
    return success_response(res)
