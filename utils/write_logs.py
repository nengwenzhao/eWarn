# coding: utf-8
from datetime import datetime
import os


def write_log(str_data):
    """
    创建txt，并且写入
    """
    now = datetime.now()
    today_time = now.strftime('%Y_%m_%d')
    path_file_name = './logs2/{}.txt'.format(today_time)
    if not os.path.exists(path_file_name):
        with open(path_file_name, "w") as f:
            print(f)

    with open(path_file_name, "a") as f:
        f.write(str_data)
