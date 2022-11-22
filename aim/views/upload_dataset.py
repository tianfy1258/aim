import os

from django.db import transaction
from aim.utils import *
from aim.models import *
from backend.settings import UPLOAD_PATH, DATA_PATH
import zipfile
import pandas as pd
import time


def upload(request):
    file_obj = request.FILES.get('file', None)
    upload_token = request.POST.get('token')
    LOGGER.debug(f"upload_token:{upload_token}\nfilename: {file_obj.name} filesize: {file_obj.size}")
    if not os.path.exists(UPLOAD_PATH):
        os.mkdir(UPLOAD_PATH)
    file_path = fr"{UPLOAD_PATH}\{upload_token}_{file_obj.name}"
    with open(file_path, 'wb') as f:
        total = file_obj.size
        chunk_size = 1024 * 1024 * 512
        for i,line in enumerate(file_obj.chunks(chunk_size=chunk_size)):
            LOGGER.debug(f"write: {(i + 1) * chunk_size} / {total}")
            f.write(line)
        f.close()
    LOGGER.info(f"文件写入: {file_path}")

    res = {}
    return success_response(res)


def after_upload_dataset(request):
    if request.method != 'POST':
        return error_response({})

    req = json.loads(request.body)
    upload_token = req["token"]
    csv_name = req['csvName']
    zip_name = req['zipName']
    zip_filepath = fr"{UPLOAD_PATH}\{upload_token}_{zip_name}"
    csv_filepath = fr"{UPLOAD_PATH}\{upload_token}_{csv_name}"

    def process(request) -> Union[str, dict]:

        req = json.loads(request.body)
        upload_token = req["token"]
        csv_name = req['csvName']
        zip_name = req['zipName']
        LOGGER.debug(f"csvName: {csv_name}  zipName: {zip_name}")

        LOGGER.debug("尝试读取zip文件")
        try:
            zip_file = zipfile.ZipFile(fr"{UPLOAD_PATH}\{upload_token}_{zip_name}", "r")
        except Exception as e:
            LOGGER.error(exc_info=True,msg=f"zip文件错误: {e}")
            return "zip文件打开失败"

        if len(zip_file.filelist) == 0:
            return "zip文件为空文件"

        LOGGER.debug("尝试读取csv")
        try:
            df = pd.read_csv(fr"{UPLOAD_PATH}\{upload_token}_{csv_name}")
        except pd.errors.EmptyDataError as e:
            LOGGER.error(exc_info=True,msg=f"csv文件错误: {e}")
            return "csv读取错误，请检查是否为空或逗号分隔"

        LOGGER.debug("检测列名是否存在")
        df_columns = df.columns.tolist()
        no_filename = "filename" if "filename" not in df_columns else None
        no_label = "label" if "label" not in df_columns else None
        hasLost = no_label is not None or no_filename is not None
        allLost = no_label is None and no_filename is None
        if hasLost:
            return f"csv文件中未发现{no_filename}{'和' if allLost else ''}{no_label}列"

        LOGGER.debug("比较csv文件和zip文件内容是否匹配")
        temp_df = df.set_index("filename")
        temp_filename_set = set(df["filename"])
        for fn in zip_file.filelist:
            if fn.filename not in temp_filename_set:
                return f"csv文件未含有{fn.filename}的标签"
            label = temp_df.loc[fn.filename]["label"]
            if label is None or label == "":
                return f"csv文件中，{fn.filename}的标签为空"

        LOGGER.debug("作简单处理，重新写入csv")
        df["filename"] = df["filename"].str.strip()
        df["label"] = df["label"].astype('str').str.strip()
        df.to_csv(fr"{UPLOAD_PATH}\{upload_token}_{csv_name}", index=None)

        tags_set = set(df["label"])

        LOGGER.debug("返回结果")
        res = {}
        res['db_instances'] = len(zip_file.filelist)
        res["db_tags_num"] = len(tags_set)
        res["db_tags"] = list(tags_set)
        res["csv_path"] = fr"{UPLOAD_PATH}\{upload_token}_{csv_name}"
        res["zip_path"] = fr"{UPLOAD_PATH}\{upload_token}_{zip_name}"

        return res

    res = process(request)

    if isinstance(res, str):
        for filepath in [zip_filepath, csv_filepath]:
            if os.path.exists(filepath):
                os.remove(filepath)
                LOGGER.info(f"删除 {filepath}")
        return error_response({}, res,duration=5000)

    return success_response(res)


@transaction.atomic
def create_dataset(request):
    if request.method != 'POST':
        return error_response({})

    req = json.loads(request.body)
    try:
        db_name = req['db_name']
        db_description = req['db_description']
        db_instances = req['db_instances']
        db_tags_num = req['db_tags_num']
        csv_path = req['csv_path']
        zip_path = req['zip_path']
        create_user = User.objects.get(user_id=request.session.get('user_id'))
    except Exception as e:
        LOGGER.error(exc_info=True,msg=f"创建数据集失败，参数错误: {e}")
        return error_response({}, "系统错误，解析参数时出现问题")
    db_filename = md5(fr"{create_user.username}_{time.time()}_{db_name}")
    db_filepath = fr"{DATA_PATH}\{db_filename}"
    try:
        r = Dataset.objects.filter(dataset_name=db_name)
        if len(r) > 0:
            LOGGER.debug("重复的数据集名称")
            return error_response({},"数据集名称已存在")

        dataset = Dataset(
            dataset_name=db_name,
            dataset_description=db_description,
            dataset_filename=db_filename,
            dataset_size=os.path.getsize(zip_path),
            dataset_instances=db_instances,
            labels_num=db_tags_num,
            create_user=create_user,
            update_user=create_user
        )
        dataset.save()
    except Exception as e:
        LOGGER.error(exc_info=True,msg=f"创建数据集失败，无法写入数据集对象 {e}")
        return error_response({}, "系统错误，无法写入数据集对象")

    try:
        import shutil
        shutil.move(zip_path, fr"{db_filepath}.zip")
        shutil.move(csv_path, fr"{db_filepath}.csv")
    except Exception as e:
        LOGGER.error(exc_info=True,msg="创建数据集失败，无法获取到上传文件")
        return error_response({}, "系统错误，无法获取到上传文件")

    return success_response({})
