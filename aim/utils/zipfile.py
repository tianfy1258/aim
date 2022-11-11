from typing import List

from backend.settings import DATA_PATH
import zipfile
from .logger import *
from .exception import *
import io
from PIL import Image
from .cache import ZipFileCache

ZIP_CACHE = ZipFileCache()


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


def extract_one_image(zfile: zipfile.ZipFile, filename: str):
    try:
        image = zfile.read(filename)
    except Exception as e:
        LOGGER.error(f"抽取文件失败：{e}")
        raise CustomException(f"抽取文件失败")
    LOGGER.debug(f"取出{filename}")
    return Image.open(io.BytesIO(image))


def extract_images(zfile: zipfile.ZipFile, filenames: List[str]):
    return [
        extract_one_image(zfile, x)
        for x in filenames
    ]
