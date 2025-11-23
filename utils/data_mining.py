import os
import random
from scipy import io as sio
import numpy as np
from torchvision import transforms

# from dataset.Dataset import get_image_names, get_text, get_label
from utils import replay_samples
# from utils.get_coco_data import get_coco_data


def load_mat(image_dir):
    """
    读取文本和标签的matlib文件
    :param image_dir:
    :return:
    """
    text_dir = os.path.join(image_dir, "tagList.mat")
    label_dir = os.path.join(image_dir, "labelList.mat")
    full_txt_dir = os.path.join(image_dir, "texts.mat")
    full_label_dir = os.path.join(image_dir, "labels.24.mat")
    all_txt = np.array(sio.loadmat(text_dir)['YAll'])
    all_label = np.array(sio.loadmat(label_dir)['LAll'])
    full_txt = np.array(sio.loadmat(full_txt_dir)['texts'])
    full_label = np.array(sio.loadmat(full_label_dir)["labels"])
    # print("Read text and label mat files：all_text,all_label,texts,labels")
    # print(
    #     f"Total Obtained Content:\n"
    #     f"Text:{all_txt.shape}  Label:{all_label.shape}  Original Text:{full_txt.shape}  Original Label:{full_label.shape}\n")
    return all_txt, all_label, full_txt, full_label


def get_index(image_dir):
    """
    获取新旧数据编号
    :param image_dir:
    :return:
    """
    new_image_dir = os.path.join(image_dir, "new_super_category.txt")
    old_image_dir = os.path.join(image_dir, "old_super_category.txt")
    new_image_name, old_image_name, new_image_idx, old_image_idx = [], [], [], []
    i, j = 0, 0
    with open(new_image_dir, "r", encoding="utf-8") as n:
        lines = n.readlines()
        for line in lines:
            new_image_idx.append(i)
            name = "im" + line + ".jpg"
            new_image_name.append(name)
            i = i + 1

    with open(old_image_dir, "r", encoding="utf-8") as n:
        lines = n.readlines()
        for line in lines:
            old_image_idx.append(j)
            name = "im" + line + ".jpg"
            old_image_name.append(name)
            j = j + 1

    new_image_name = np.array(new_image_name)
    old_image_name = np.array(old_image_name)
    print("Obtain image names and image indexes from old and new datasets：\n"
          f"Number of new and old image names：{len(new_image_name)},{len(old_image_name)}\n，Number of new and old indexes：{len(new_image_idx)},{len(old_image_idx)}")
    return new_image_name, old_image_name, new_image_idx, old_image_idx


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
    return id_clean_2


def get_random_index(id_clean_2, train_nums, query_nums, database_nums):
    """
    使用处理过的数据集来划分训练集，查询集和数据库的索引
    :param id_clean_2:
    :param train_nums:
    :param query_nums:
    :param database_nums:
    :return:
    """
    sample_list = [i for i in range(len(id_clean_2))]
    train_sample_list = random.sample(sample_list, train_nums)
    for i in train_sample_list:
        sample_list.remove(i)
    query_sample_list = random.sample(sample_list, query_nums)
    for i in query_sample_list:
        sample_list.remove(i)
    database_sample_list = random.sample(sample_list, database_nums)
    print(
        f"Number of training set samples：{len(train_sample_list)}\n"
        f"Number of query set samples：{len(query_sample_list)}\n"
        f"Number of Dataset Samples:{len(database_sample_list)}\n")
    return train_sample_list, query_sample_list, database_sample_list


def get_nuw_old(image_dir):
    """
    读取新类和旧类的数据
    :param image_dir:
    :return:
    """
    num_new_2 = []
    num_old_2 = []
    with open(os.path.join(image_dir, "new_super_category.txt"), "r", encoding="utf-8") as f:
        for line in f:
            num_new_2.append(int(line.split("/n")[0]))
    with open(os.path.join(image_dir, "old_super_category.txt"), "r", encoding="utf-8") as f:
        for line in f:
            num_old_2.append(int(line.split("/n")[0]))
    print(f"Number of new class samples:{len(num_new_2)}\n"
          f"Number of old class samples:{len(num_old_2)}")
    return num_new_2, num_old_2


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset0 = int(n_total * ratio)
    offset1 = int(n_total * (1 - ratio))

    if n_total == 0:  # 列表为空的情况
        return []

    if offset0 + offset1 > n_total:  # 错误切分条件
        print("Incorrect segmentation ratio!!!")
        return 0

    if offset0 + offset1 <= n_total:  # 切分
        random.shuffle(full_list)
    sublist_1 = full_list[:offset0]
    sublist_2 = full_list[offset0:offset0 + offset1]
    return sublist_1, sublist_2


def split_sample(image_dir, id_clean_2, train_sample_list, query_sample_list, database_sample_list, seen_num, all_num):
    """
    将得到的索引根据新旧类的数据，划分为新旧数据
    :param image_dir:
    :param id_clean_2:
    :param train_sample_list:
    :param query_sample_list:
    :param database_sample_list:
    :return:
    """
    count = 0
    num_new_2, num_old_2 = get_nuw_old(image_dir)
    train_new_list, train_old_list, query_new_list, query_old_list, database_new_list, database_old_list = [], [], [], [], [], []
    for i in train_sample_list:
        if id_clean_2[i] in num_new_2:
            train_new_list.append(i)
        else:
            train_old_list.append(i)
        count = count + 1
    count = 0
    for i in query_sample_list:
        if id_clean_2[i] in num_new_2:
            query_new_list.append(i)
        else:
            query_old_list.append(i)
        count = count + 1
    count = 0
    for i in database_sample_list:
        if id_clean_2[i] in num_new_2:
            database_new_list.append(i)
        else:
            database_old_list.append(i)
        count = count + 1
    train_sample_list = (train_new_list, train_old_list)
    query_sample_list = (query_new_list, query_old_list)
    database_sample_list = (database_new_list, database_old_list)
    ratio = 0.2
    sub_train_data_1, _ = data_split(train_old_list, ratio, shuffle=True)
    train_extensible_list = train_new_list + sub_train_data_1
    replay_samples_indices = replay_samples.get_replay_samples("mirflickr", seen_num, all_num)
    print(
        f"Number of training set samples:{len(train_sample_list[0] + train_sample_list[1])}\n"
        f"Number of old class samples in the training set:{len(train_sample_list[0])}\n"
        f"Number of new class samples in the training set:{len(train_sample_list[1])}\n"
        f"Number of query set samples:{len(query_sample_list[0] + query_sample_list[1])}\n"
        f"Number of old class samples in the query set:{len(query_sample_list[0])}\n"
        f"Number of new class samples in the query set:{len(query_sample_list[1])}\n"
        f"Number of database samples:{len(database_sample_list[0] + database_sample_list[1])}\n"
        f"Number of old class samples in the database:{len(database_sample_list[0])}\n"
        f"Number of new class samples in the database:{len(database_sample_list[1])}\n")
    print(f"Number of samples for replaying data：{len(replay_samples_indices)}")
    return train_sample_list, query_sample_list, database_sample_list, train_extensible_list, replay_samples_indices


def get_image_names(id_clean_2, sample_list):
    image_names = []
    for i in sample_list:
        image_names.append("im" + str(id_clean_2[i]) + ".jpg")
    return image_names


def get_image_names2(sample_list):
    image_names = []
    for i in sample_list:
        image_names.append("im" + str(i) + ".jpg")
    return image_names


def get_text(all_text, sample_list):
    text = []
    for i in sample_list:
        text.append(all_text[i])
    return text


def get_text2(all_text, sample_list):
    text = []
    for i in sample_list:
        text.append(all_text[i - 1])
    return text


def get_label(all_label, sample_list):
    label = []
    for i in sample_list:
        label.append(all_label[i])
    return label


def get_label2(all_label, sample_list):
    label = []
    for i in sample_list:
        label.append(all_label[i - 1])
    return label


def get_database_data(id_clean_2, database_sample_index, all_label, all_text):
    database_names, database_text, database_label = [], [], []
    print(database_sample_index)
    for i in database_sample_index:
        database_names.append("im" + str(id_clean_2[i]) + ".jpg")
        database_text.append(all_text[i])
        database_label.append(all_label[i])
    return database_names, database_text, database_label


def get_original_index(id_clean_2, replay_samples_index):
    original_index = []
    count = 0
    for index in replay_samples_index:
        while True:
            if id_clean_2[count] == index:
                original_index.append(count)
                count = 0
                break
            else:
                count = count + 1
    return original_index


def split_data(image_dir, all_text, all_label, full_text, seen_num, all_num):
    """
    划分数据集
    :param image_dir:
    :param train_nums:
    :param query_nums:
    :param database_nums:
    :param all_text:
    :param all_label:
    :return:
    """
    print("Partitioning data...")
    train_nums = 12015
    query_nums = 6000
    database_nums = 2000
    id_clean_2 = get_clean_id(image_dir)
    train_sample_index, query_sample_index, database_sample_index = get_random_index(id_clean_2, train_nums, query_nums,
                                                                                     database_nums)
    database_names_all, database_text_all, database_label_all = get_database_data(id_clean_2, database_sample_index,
                                                                                  all_label, all_text)
    train_sample_index, query_sample_index, database_sample_index, train_extensible_index, replay_sample_index = split_sample(
        image_dir, id_clean_2, train_sample_index, query_sample_index, database_sample_index, seen_num, all_num)
    replay_sample_index = get_original_index(id_clean_2, replay_sample_index)
    train_image_names = (
        get_image_names(id_clean_2, train_sample_index[0]),
        get_image_names(id_clean_2, train_sample_index[1]))
    query_image_names = (
        get_image_names(id_clean_2, query_sample_index[0]),
        get_image_names(id_clean_2, query_sample_index[1]))
    database_image_names = (
        get_image_names(id_clean_2, database_sample_index[0]),
        get_image_names(id_clean_2, database_sample_index[1]))
    train_extensible_names = get_image_names(id_clean_2, train_extensible_index)
    replay_image_names = get_image_names(id_clean_2, replay_sample_index)

    train_text = (get_text(all_text, train_sample_index[0]), get_text(all_text, train_sample_index[1]))
    query_text = (get_text(all_text, query_sample_index[0]), get_text(all_text, query_sample_index[1]))
    database_text = (get_text(all_text, database_sample_index[0]), get_text(all_text, database_sample_index[1]))
    train_extensible_text = get_text(all_text, train_extensible_index)
    replay_text = get_text(all_text, replay_sample_index)

    train_label = (get_label(all_label, train_sample_index[0]), get_label(all_label, train_sample_index[1]))
    query_label = (get_label(all_label, query_sample_index[0]), get_label(all_label, query_sample_index[1]))
    database_label = (get_label(all_label, database_sample_index[0]), get_label(all_label, database_sample_index[1]))
    train_extensible_label = get_label(all_label, train_extensible_index)
    replay_label = get_label(all_label, replay_sample_index)

    image_names = (
        train_image_names, query_image_names, database_image_names, train_extensible_names, replay_image_names)
    text = (train_text, query_text, database_text, train_extensible_text, replay_text)
    label = (train_label, query_label, database_label, train_extensible_label, replay_label)
    print("Partition data completed!!！\n")
    return image_names, text, label, database_names_all, database_text_all, database_label_all


def get_image_transform():
    resize = [224, 224]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), normalize_transform])
    return transform


def get_coco_transform():
    resize = [224, 224]
    normal_mean = [103.530, 116.280, 123.675]
    normal_std = [57.375, 57.120, 58.395]
    normalize_transform = transforms.Normalize(normal_mean, normal_std)
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), normalize_transform])
    return transform