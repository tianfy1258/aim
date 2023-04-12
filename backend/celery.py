import os
from celery import Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

app = Celery('backend',broker='redis://127.0.0.1:6379/0')
app.config_from_object('django.conf:settings', namespace='CELERY')
# 使用CELERY_ 作为前缀，在settings中写配置

# 发现任务文件每个app下的task.py
app.autodiscover_tasks(['aim.views.measurement.model_task','aim.views.measurement.dataset_task'])

# 指令启动方案
# 命令行 进入项目目录启动worker:
#
# celery -A backend worker -l info -P eventlet --concurrency=3 --pool=solo

# 周期任务
# celery -A IntelligentSystemPlatformBackend beat -l info -S django
# 更改后可能需要python manage.py migrate
# 上面两个都得启动
