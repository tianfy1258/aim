# only used for generating fake data

from aim.models import *
from faker import Faker
fk = Faker(locale="zh-CN")
import random as r
li = []
count = 2567
for i in range(count):
    intances = r.randint(0, 100000)
    size = intances * r.randint(10_000, 5_000_000) * min(0.5, r.random())
    dataset = Dataset(
        dataset_name=fk.name(),
        dataset_description=fk.text(),
        dataset_filename="null",
        dataset_size=size,
        dataset_instances=intances,
        labels_num=r.randint(1, 1001),
        create_user=User.objects.get(user_id=1),
        update_user=User.objects.get(user_id=1)
    )
    li.append(dataset)

Dataset.objects.bulk_create(li)
