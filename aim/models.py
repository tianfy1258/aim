from django.db import models


class User(models.Model):
    # 用户id (主键)
    user_id = models.AutoField(primary_key=True)
    # 用户名 (唯一)
    username = models.CharField(max_length=255, unique=True)
    # 密码
    password = models.CharField(max_length=255, default='123456')
    # 用户权限 [user, manager]
    user_auth = models.CharField(max_length=255, default='user')
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 修改时间
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True,
        db_table = "user"


# 数据集表
class Dataset(models.Model):
    # 数据集id (主键)
    dataset_id = models.AutoField(primary_key=True)
    # 数据集名称 (唯一)
    dataset_name = models.CharField(max_length=255)
    # 数据集描述
    dataset_description = models.CharField(max_length=255, null=True)
    # 数据集文件名称
    dataset_filename = models.CharField(max_length=255)
    # 数据集大小 [bytes]
    dataset_size = models.BigIntegerField()
    # 图片数量
    dataset_instances = models.IntegerField()
    # 标签数量
    labels_num = models.IntegerField()
    # 数据集图片名称的hashcode
    hashcode = models.CharField(max_length=255)
    # 数据类型 [没用，留着拓展]
    dataset_type = models.CharField(max_length=255, default='images')

    # 创建者
    create_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name='dataset_create_user')
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 修改时间
    update_time = models.DateTimeField(auto_now=True)
    # 修改人
    update_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name='dataset_update_user')

    class Meta:
        managed = True,
        db_table = "dataset"


# 配置表
class Configuration(models.Model):
    # 配置id (主键)
    config_id = models.AutoField(primary_key=True)
    # 配置名称 (唯一)
    config_name = models.CharField(max_length=255)
    # 配置描述
    config_description = models.CharField(max_length=255, null=True)
    # 配置内容 [json格式存储]
    config_content = models.CharField(max_length=255)
    # 配置类型 [没用，留着扩展]
    config_type = models.CharField(max_length=255, default='interpreter')

    # 创建人
    create_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name='config_create_user')
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 修改人
    update_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name='config_update_user')
    # 修改时间
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True,
        db_table = "configuration"


# # 覆盖测试任务表
# class CoverageTask(models.Model):
#     # 任务id (主键)+
#     task_id = models.AutoField(primary_key=True)
#     # 用户id (外键)
#     user = models.ForeignKey("User", on_delete=models.CASCADE)
#     # 任务名
#     task_name = models.CharField(max_length=255)
#     # 任务描述
#     task_description = models.CharField(max_length=255, null=True)
#     # 任务配置 [json存储]
#     task_config = models.CharField(max_length=255)
#     # 任务状态 [fail, running, terminated,canceled, success]
#     task_status = models.CharField(max_length=255)
#     # 任务状态说明
#     task_status_description = models.CharField(max_length=255, default='')
#
#     # 创建者
#     create_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="coverage_create_user")
#     # 创建时间
#     create_time = models.DateTimeField(auto_now_add=True)
#     # 修改者
#     update_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="coverage_update_user")
#     # 修改时间
#     update_time = models.DateTimeField(auto_now=True)
#
#     class Meta:
#         managed = True,
#         db_table = "coverageTask"


# 模型度量
class ModelMeasurementTask(models.Model):
    # 任务id (主键)+
    task_id = models.CharField(max_length=255, primary_key=True)
    # 任务名
    task_name = models.CharField(max_length=255)
    # 任务描述
    task_description = models.CharField(max_length=255, null=True)
    # 数据集
    dataset_id = models.ForeignKey("Dataset", on_delete=models.CASCADE)
    # 样本数
    sample_count = models.IntegerField()
    # 是否随机数种子
    enable_random = models.BooleanField(default=False)
    # 随机数种子
    random_seed = models.IntegerField(default=0)
    # 测试模型
    model_id = models.ForeignKey("DeepModel", on_delete=models.CASCADE)
    # 度量方法
    measure_method = models.CharField(max_length=255, default='')
    # 任务状态
    task_status = models.CharField(max_length=255, default='PENDING')
    # 任务结果
    task_result = models.CharField(max_length=255, default='')
    # 错误日志
    task_traceback = models.CharField(max_length=2048, default='')
    # 创建者
    create_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="model_measurement_create_user")
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 修改者
    update_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="model_measurement_update_user")
    # 修改时间
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True,
        db_table = "modelMeasurementTask"


# 数据质量度量任务
class DatasetMeasurementTask(models.Model):
    # 任务id (主键)+
    task_id = models.CharField(max_length=255, primary_key=True)
    # 任务名
    task_name = models.CharField(max_length=255)
    # 任务描述
    task_description = models.CharField(max_length=255, null=True)
    # 数据集
    dataset_id = models.ForeignKey("Dataset", on_delete=models.CASCADE)
    # 是否对比分析
    enable_compare = models.BooleanField(default=False)
    # 对比数据集 [待开发]
    # dataset_compare_id = models.ForeignKey("Dataset", on_delete=models.CASCADE,null=True,related_name="dataset_compare_id")
    # 样本数
    sample_count = models.IntegerField()
    # 是否随机数种子
    enable_random = models.BooleanField(default=False)
    # 随机数种子
    random_seed = models.IntegerField(default=0)
    # 度量方法
    single_measure_method = models.CharField(max_length=255, default='')
    # 对比方法
    compare_measure_method = models.CharField(max_length=255, default='')
    # 任务状态
    task_status = models.CharField(max_length=255, default='PENDING')
    # 任务结果
    task_result = models.CharField(max_length=255, default='')
    # 错误日志
    task_traceback = models.CharField(max_length=2048, default='')

    # 创建者
    create_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="dataset_measurement_create_user")
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 修改者
    update_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="dataset_measurement_update_user")
    # 修改时间
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True,
        db_table = "datasetMeasurementTask"


# 模型表
class DeepModel(models.Model):
    # 模型id (主键)
    model_id = models.AutoField(primary_key=True)
    # 模型名
    model_name = models.CharField(max_length=255)
    # 模型描述
    model_description = models.CharField(max_length=255, null=True)
    # 模型训练数据集
    model_dataset = models.CharField(max_length=255, null=True)
    # 是否为预定义模型
    is_predefine = models.BooleanField(default=False)
    # 是否为从方法中获取模型
    is_use_function = models.BooleanField(default=False)
    # 模型大小
    model_size = models.BigIntegerField()
    # 模型类别 [LeNet、AlexNet ...]
    model_type = models.CharField(max_length=255, default="LeNet")
    # 模型分类类别个数
    model_output_shape = models.IntegerField()
    # 模型文件名 [模型二进制文件pt, 模型定义文件py, 输出映射文件csv]
    model_filename = models.CharField(max_length=255)
    # 模型文件类名
    model_classname = models.CharField(max_length=255)
    # 模型图片处理方法名称
    model_processor = models.CharField(max_length=255)

    # 创建者
    create_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="model_create_user")
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 修改者
    update_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="model_update_user")
    # 修改时间
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True,
        db_table = "model"
