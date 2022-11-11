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


# 任务表
class Task(models.Model):
    # 任务id (主键)+
    task_id = models.AutoField(primary_key=True)
    # 用户id (外键)
    user = models.ForeignKey("User", on_delete=models.CASCADE)
    # 任务名
    task_name = models.CharField(max_length=255)
    # 任务描述
    task_description = models.CharField(max_length=255, null=True)
    # 任务配置 [json存储]
    task_config = models.CharField(max_length=255)
    # 任务状态 [fail, running, canceled, success]
    task_status = models.CharField(max_length=255)
    # 任务状态说明
    task_status_description = models.CharField(max_length=255, default='')

    # 创建者
    create_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="task_create_user")
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True)
    # 修改者
    update_user = models.ForeignKey("User", on_delete=models.CASCADE, related_name="task_update_user")
    # 修改时间
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True,
        db_table = "task"


# 模型表
class DeepModel(models.Model):
    # 模型id (主键)
    model_id = models.AutoField(primary_key=True)
    # 模型名
    model_name = models.CharField(max_length=255)
    # 模型描述
    model_description = models.CharField(max_length=255, null=True)
    # 模型大小
    model_size = models.BigIntegerField()
    # 模型类别 [LeNet、AlexNet ...]
    model_type = models.CharField(max_length=255,default="LeNet")
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
