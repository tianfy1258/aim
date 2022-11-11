from .main import IMAGE_CACHE
from PIL import Image
import io
from django.http import HttpResponse

def get_image(request):
    """
    通过url获得缓存中的图片
    :param request:
    :return:
    """
    key = request.GET.get("url")
    img: Image = IMAGE_CACHE.get(key, None)
    buf = io.BytesIO()
    img.save(buf,format='png')
    buf.seek(0)
    return HttpResponse(buf, content_type='image/jpg')
