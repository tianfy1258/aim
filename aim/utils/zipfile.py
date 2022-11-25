from typing import List

from backend.settings import DATA_PATH
import zipfile
from .logger import *
from .exception import *
import io
from PIL import Image
from .cache import ZipFileCache, ZipImageFileCache

ZIP_CACHE = ZipFileCache()
IMAGE_CACHE = ZipImageFileCache()


def get_zipfile(dataset_name: str):
    dataset_path = fr"{DATA_PATH}\{dataset_name}.zip"
    try:
        zfile = ZIP_CACHE.get(dataset_path, None)
        if zfile is None:
            zfile = zipfile.ZipFile(dataset_path)
            LOGGER.debug(f"打开新的zip对象:{dataset_path}")
            ZIP_CACHE.set(dataset_path, zfile)
    except Exception as e:
        LOGGER.error(f"打开数据集文件失败: {e}", exc_info=True)
        raise CustomException("打开数据集文件失败")

    return zfile


def extract_one_image(zfile: zipfile.ZipFile, filename: str, use_cache=True):
    try:
        if use_cache:
            image = IMAGE_CACHE.get(f"{id(zfile)}{filename}", None)
            if image is None:
                image = zfile.read(filename)
                IMAGE_CACHE.set(f"{id(zfile)}{filename}", image)
        else:
            image = zfile.read(filename)
    except Exception as e:
        LOGGER.error(f"抽取文件失败：{e}")
        raise CustomException(f"抽取文件失败")
    return Image.open(io.BytesIO(image)).convert("RGB")


def extract_images(zfile: zipfile.ZipFile, filenames: List[str], use_cache=True):
    for x in filenames:
        yield extract_one_image(zfile, x, use_cache)
