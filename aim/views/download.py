from aim.models import *
from aim.utils import *
from django.http import FileResponse


def dataset_download(request):
    dataset_id = request.GET["id"]
    try:
        dataset = Dataset.objects.get(dataset_id=dataset_id)
    except Exception as e:
        LOGGER.error(exc_info=True,msg=f"数据库读取失败: {e}")
        return error_response({}, "下载失败，数据集信息未在数据库中")

    filepath = fr"{DATA_PATH}\{dataset.dataset_filename}.zip"
    try:
        def file_iter(fp, chunk_size=1024):
            with open(fp, 'rb') as f:
                while True:
                    c = f.read(chunk_size)
                    if c:
                        yield c
                    else:
                        break
    except Exception as e:
        LOGGER.error(exc_info=True,msg=f"文件读取失败: {e}")
        return error_response({}, "下载失败，未找到该数据集")

    response = FileResponse(file_iter(filepath), filename=dataset.dataset_name)
    response['Content-Disposition'] = f'attachment; filename={dataset.dataset_name}.zip'
    return response
