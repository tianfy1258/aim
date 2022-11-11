"""
Django settings for backend project.

Generated by 'django-admin startproject' using Django 3.2.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.2/ref/settings/
"""
import os
from pathlib import Path
from configparser import ConfigParser
import matplotlib
matplotlib.use('agg')
print("matplotlib.backend: " + matplotlib.get_backend())

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')
DATABASE_CONFIG = CONFIG_PARSER["database"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_PATH = BASE_DIR + r"\upload"
DATA_PATH = BASE_DIR + r"\dataset"
MODEL_PATH = BASE_DIR + r"\deep_model"
for path in [UPLOAD_PATH, DATA_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-_abnbo*%!8%-ry05v_k%tft3c&u5-r)0%+l(d(05#tyu-4m919'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
SILK = False
# SILKY_PYTHON_PROFILER = True
ALLOWED_HOSTS = ['*']

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    # 'silk',
    'aim.apps.AimConfig',
]
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # 'silk.middleware.SilkyMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'dist']
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': DATABASE_CONFIG["db_name"],
        'USER': DATABASE_CONFIG["db_user"],
        'PASSWORD': DATABASE_CONFIG["db_password"],
        'HOST': DATABASE_CONFIG["db_host"],
        'PORT': DATABASE_CONFIG["db_port"]
    },
}
# CACHES = {
#     "default": {
#         "BACKEND": "django_redis.cache.RedisCache",
#         # 使用的redis数据
#         "LOCATION": "redis://127.0.0.1:6379",
#         "OPTIONS": {
#             "CLIENT_CLASS": "django_redis.client.DefaultClient",
#             # 连接池最大数值
#             "CONNECTION_KWARGS": {"max_connection": 100},
#             # "PASSWORD":'12334'redis的密码
#         }
#     }
# }

# Password validation
# https://docs.djangoproject.com/en/3.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'zh-hans'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True

USE_TZ = False

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.0/howto/static-files/

STATIC_URL = '/assets/'

# 前端路径
FRONTEND_ROOT = 'dist'
# 模块资源引用地址
STATICFILES_DIRS = (
    os.path.join(BASE_DIR, FRONTEND_ROOT),
    os.path.join(BASE_DIR, FRONTEND_ROOT + '/assets/'),
    os.path.join(BASE_DIR, 'deep_model/'),
    os.path.join(BASE_DIR, 'dataset/'),
)

# 增加跨域忽略
CORS_ORIGIN_ALLOW_ORIGINS = True
CORS_ORIGIN_ALLOW_ALL = True
# 允许带cookie
CORS_ALLOW_CREDENTIALS = True

# 设置允许访问的方法
CORS_ALLOW_METHODS = (
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
    'VIEW',
)
# 设置允许的header
CORS_ALLOW_HEADERS = (
    'XMLHttpRequest',
    'X_FILENAME',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'Pragma',
    'token'
)

# 输出sql语句
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        # 详细的日志格式
        'standard': {
            'format': '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]'
                      '[%(levelname)s][%(message)s]'
        },
        # 简单的日志格式
        'simple': {
            'format': '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s'
        },

    },

    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'django.db.backends': {
            'handlers': ['console'],
            'propagate': True,
            'level': 'DEBUG',
        },
        'custom': {
            'handlers': ['console'],
            'propagate': True,
            'level': 'DEBUG',
        }
    }
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
