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
        elif attribution_method == 'Saliency':
            attribution = Saliency(req)
        elif attribution_method == 'DeepLift':
            attribution = DeepLift(req)
        elif attribution_method == "Occlusion":
            attribution = Occlusion(req)

        res = {
            "data": attribution.attribute_from_dataset()
        }
    except CustomException as e:
        LOGGER.error("自定义错误", exc_info=True)
        return error_response({}, str(e), duration=5000)
    except Exception as e:
        LOGGER.error(e, exc_info=True)
        return error_response({}, f"分析时出现错误：{e}", duration=5000)
    return success_response(res)
