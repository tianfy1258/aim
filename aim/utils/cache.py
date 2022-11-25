from django.core.cache import cache
import time
from aim.utils import md5, LOGGER

"""

1.ZIP对象不能用缓存，报错 cannot serialize '_io.BufferedReader' object，这里单独适配下，就简单return
2.BaseCoverage对象，这里也用dict，因为缓存的不是静态对象，而是动态的



"""


class ZipFileCache:
    def __init__(self):
        self.key = "zipfile_"
        self.cache = {}
        LOGGER.info("ZipFileCache: 创建dict引用缓存")

    def set(self, key, value, expired_time=600):
        LOGGER.debug(f"缓存：⭕ 存入zip对象:{key}")
        self.cache[fr"{self.key}{key}"] = value

    def get(self, key, defult_value=None):
        r = self.cache.get(fr"{self.key}{key}", defult_value)
        if r is not None:
            LOGGER.debug(f"缓存：✔ 找到缓存的zip对象:{key}")
        else:
            LOGGER.debug(f"缓存：❌ 未找到zip对象:{key}")
        return r


class CoverageCache:
    def __init__(self):
        self.key = "coverage_"
        self.cache = {}
        LOGGER.info("CoverageCache: 创建dict引用缓存")

    def set(self, key, value):
        LOGGER.debug(f"缓存：⭕ 存入覆盖测试对象:{key}")
        self.cache[fr"{self.key}{key}"] = value

    def get(self, key, defult_value=None):
        r = self.cache.get(fr"{self.key}{key}", defult_value)
        if r is not None:
            LOGGER.debug(f"缓存：✔ 找到缓存的覆盖测试对象:{key}")
        else:
            LOGGER.debug(f"缓存：❌ 未找到覆盖测试对象:{key}")
        return r

    def remove(self, key):
        del self.cache[fr"{self.key}{key}"]
        LOGGER.debug(f"缓存：✔ 删除缓存的覆盖测试对象:{key}")


class ImageFileCache:
    def __init__(self):
        self.key = "imagefile_"

    def set(self, key, value, expired_time=7200):
        true_filename = f"{md5(key + str(time.time()))}"
        LOGGER.debug(f"缓存：⭕ 存入图片文件:{true_filename}")
        cache.set(fr"{self.key}{true_filename}", value, expired_time)
        return true_filename

    def get(self, key, defult_value=None):
        r = cache.get(fr"{self.key}{key}", defult_value)
        if r is not None:
            LOGGER.debug(f"缓存：✔ 找到缓存的图片对象:{key}")
            LOGGER.debug(f"缓存：✔ 更新图片对象缓存时间:7200s")
            cache.touch(key, 7200)
        else:
            LOGGER.debug(f"缓存：❌ 未找到图片对象:{key}")
        return r


class ZipImageFileCache:
    def __init__(self):
        self.key = "zipimagefile_"

    def set(self, key, value, expired_time=600):
        key = f"{self.key}{key}"
        LOGGER.debug(f"缓存：⭕ 存入图片文件:{key}")
        cache.set(key, value, expired_time)

    def get(self, key, defult_value=None):
        r = cache.get(fr"{self.key}{key}", defult_value)
        if r is not None:
            LOGGER.debug(f"缓存：✔ 找到缓存的图片对象:{key}")
            LOGGER.debug(f"缓存：✔ 更新图片对象缓存时间:600s")
            cache.touch(key, 600)
        else:
            LOGGER.debug(f"缓存：❌ 未找到图片对象:{key}")
        return r


class DeepModelCache:
    def __init__(self):
        self.key = "deepmodel_"

    def set(self, key, value, expired_time=600):
        LOGGER.debug(f"缓存：⭕ 存入模型文件:{key}")
        cache.set(fr"{self.key}{key}", value, expired_time)

    def get(self, key, defult_value=None):
        r = cache.get(fr"{self.key}{key}", defult_value)
        if r is not None:
            LOGGER.debug(f"缓存：✔ 找到缓存的模型对象:{key}")
            LOGGER.debug(f"缓存：✔ 更新模型对象缓存时间:600s")
            cache.touch(key, 600)
        else:
            LOGGER.debug(f"缓存：❌ 未找到模型对象:{key}")
        return r


class AttributionCache:
    def __init__(self):
        self.key = "attribution_"

    def set(self, key, value, expired_time=600):
        LOGGER.debug(f"缓存：⭕ 存入归因对象:{key}")
        cache.set(fr"{self.key}{key}", value, expired_time)

    def get(self, key, defult_value=None):
        r = cache.get(fr"{self.key}{key}", defult_value)
        if r is not None:
            LOGGER.debug(f"缓存：✔ 找到缓存的归因对象:{key}")
            LOGGER.debug(f"缓存：✔ 更新归因对象缓存时间:600s")
            cache.touch(key, 600)
        else:
            LOGGER.debug(f"缓存：❌ 未找到归因对象:{key}")
        return r


class AttributeResultCache:
    def __init__(self):
        self.key = "attribute_result_"

    def set(self, key, value, expired_time=7200):
        LOGGER.debug(f"缓存：⭕ 存入归因结果:{key}")
        cache.set(fr"{self.key}{key}", value, expired_time)

    def get(self, key, defult_value=None):
        r = cache.get(fr"{self.key}{key}", defult_value)
        if r is not None:
            LOGGER.debug(f"缓存：✔ 找到缓存的归因结果:{key}")
            LOGGER.debug(f"缓存：✔ 更新归因结果缓存时间:7200s")
            cache.touch(key, 7200)
        else:
            LOGGER.debug(f"缓存：❌ 未找到归因结果:{key}")
        return r
