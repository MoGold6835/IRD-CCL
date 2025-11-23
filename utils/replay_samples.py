import os
import random
import math

image_dir = "/home/carry6/LS_project/dataset/mirflickr"
file_path = os.path.join(image_dir, "mirflickr25k_annotations_v080")

def get_clean_id(image_dir):
    """
    获取处理后的数据集索引
    :param image_dir:
    :return:
    """
    id_clean_2 = []
    with open(os.path.join(image_dir, "id_clean_2.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            id_clean_2.append(int(line.split("/n")[0]))
    print(len(id_clean_2))

    return id_clean_2



def get_replay_samples(dataset_name, seen_num, all_num):
    mem_size = 2000
    id_clean_2 = get_clean_id(image_dir)
    if dataset_name == "mirflickr":
        class_list = []
        class_indices = {}
        path_list = [f for f in os.listdir(file_path) if 'README' not in f]
        for path in path_list:
            temp = path.split(".")
            if "_r1" not in temp[0]:
                class_list.append(temp[0])
                with open(os.path.join(file_path, path), "r") as file:
                    lines = file.readlines()
                    idx_list = []
                    for line in lines:
                        idx_list.append(int(line.strip()))
                class_indices[temp[0]] = idx_list
        new_class = ["people", "sky"]
        old_class_name = []
        # 获取旧类和新类的索引
        old_indices, new_indices = [], []
        for name in class_list:
            if name in new_class:
                new_indices = new_indices + class_indices[name]
            else:
                old_indices = old_indices + class_indices[name]
                old_class_name.append(name)
        class_samples_num = math.ceil(mem_size / seen_num)
        # 获取clean_indices
        replay_samples_indices = []
        count = 0
        temp_class_indices = []
        for class_name in class_list:
            if count < seen_num:
                for index in class_indices[class_name]:
                    if index in id_clean_2:
                        if index not in new_indices:
                            temp_class_indices.append(index)
                if len(temp_class_indices) <= class_samples_num:
                    replay_samples_indices = replay_samples_indices + temp_class_indices
                else:
                    replay_samples_indices = replay_samples_indices + random.sample(temp_class_indices,
                                                                                    class_samples_num)
            else:
                break
            count = count + 1
    return replay_samples_indices
