from aim.utils import *
from aim.models import *
from .main import *
from .utils import NC
import random

COVERAGE_CACHE = CoverageCache()



def coverage(request):
    if request.method != 'POST':
        return error_response({})
    req = json.loads(request.body)
    coverage_method = req['coverage']
    task_key = req['task_key']
    base_coverage: BaseCoverage = None
    LOGGER.info(task_key)
    try:
        base_coverage = COVERAGE_CACHE.get(task_key, None)
        if base_coverage is None:
            if coverage_method == 'Neuron Coverage':
                base_coverage = NeuronCoverage(req)
            elif coverage_method == '???':
                pass
            COVERAGE_CACHE.set(task_key, base_coverage)
        res = {
            "data": base_coverage.coverage_from_dataset()
        }
    except CustomException as e:
        LOGGER.error("自定义错误", exc_info=True)
        return error_response({}, str(e), duration=5000)
    except Exception as e:
        LOGGER.error(e, exc_info=True)
        return error_response({}, f"分析时出现错误：{e}", duration=5000)
    return success_response(res)


status_dict = {}


def get_status(request):
    if request.method != 'GET':
        return error_response({})
    task_key = request.GET.get("task_key")
    current_index = int(request.GET.get("current_index"))
    base_coverage: BaseCoverage = COVERAGE_CACHE.get(task_key, None)
    if base_coverage is None:
        v = status_dict.get(task_key, 0)
        status_dict[task_key] = v + 1
        LOGGER.warn(f"重新获取状态! retry : {v} / 5")
        if status_dict[task_key] == 5:
            del status_dict[task_key]
            return error_response({}, error_message="任务状态获取失败！")
        return retry_response({})
    if base_coverage.is_finished():
        COVERAGE_CACHE.remove(task_key)
    return success_response({
        "status": base_coverage.is_finished(),
        "process": len(base_coverage.result),
        "total": base_coverage.sample_count,
        "data": [x[2] for x in base_coverage.result[current_index:]]
    })
