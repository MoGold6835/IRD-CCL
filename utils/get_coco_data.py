import json
import math
import os
import random

import torch

base_dir = "/home/carry6/LS_project/dataset/coco"
train_num = 14624
query_num = 5000
database_num = 15000


def get_coco_data(base_dir):
    annotation_dir = os.path.join(base_dir, "annotations_trainval2014", "annotations")
    coco_preprocess_data = os.path.join(annotation_dir, "coco_preprocess_data.json")
    coco_vector = os.path.join(annotation_dir, "coco_textvector_data.json")

    with open(os.path.join(coco_preprocess_data), "r", encoding="utf-8") as f:
        data = f.read()
        json_data = json.loads(data)
    # print(json_data)
    class_id = json_data["class_id"]
    train_data = json_data["train_data"]
    val_data = json_data["val_data"]

    class_id_reverse = {value: key for key, value in class_id.items()}

    id_num = class_id_reverse["person"]
    person_list_train = list(set(train_data[2][id_num]))
    person_list_val = list(set(val_data[2][id_num]))

    all_train_index, all_val_index = [], []
    for key, value in train_data[2].items():
        if key != id_num:
            all_train_index = all_train_index + value
    for key, value in val_data[2].items():
        if key != id_num:
            all_val_index = all_val_index + value
    all_train_index = list(set(all_train_index).difference(set(person_list_train)))
    all_val_index = list(set(all_val_index).difference(person_list_val))

    with open(os.path.join(coco_vector), "r", encoding="utf-8") as f:
        data = f.read()
        vector_data = json.loads(data)

    animal_class = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
    jiadian_class = ["microwave", "oven", "toaster", "refrigerator", "sink"]
    indoor_class, new_id_num = ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"], []
    new_class = animal_class + jiadian_class + indoor_class
    for class_name in new_class:
        new_id_num.append(class_id_reverse[class_name])
    # print(f"新类别id：{new_id_num}")

    new_index_train, new_index_val = [], []
    for key, value in train_data[2].items():
        if key in new_id_num:
            new_index_train = new_index_train + value

    for key, value in val_data[2].items():
        if key in new_id_num:
            new_index_val = new_index_val + value
    new_index_train = list(set(new_index_train).difference(set(person_list_train)))
    new_index_val = list(set(new_index_val).difference(set(person_list_val)))

    all_text_train = vector_data["train"]
    all_text_val = vector_data["val"]

    old_train_text, new_train_text, old_query_text, new_query_text, old_database_text, new_database_text = [], [], [], [], [], []

    all_image_name = train_data[0]
    old_train_image_name, new_train_image_name, old_query_image_name, new_query_image_name, old_database_image_name, new_database_image_name = [], [], [], [], [], []

    json_dir = os.path.join(annotation_dir, "coco_label_data.json")

    with open(json_dir, "r", encoding="utf-8") as f:
        data = f.read()
    json_data = json.loads(data)
    train_dict = json_data["train"]
    old_train_label, new_train_label, old_query_label, new_query_label, old_database_label, new_database_label = [], [], [], [], [], []
    class_number = 80

    label_keys_train = list(train_dict.keys())

    temp2 = all_train_index
    temp2 = list(map(str, temp2))
    temp2 = set(temp2).difference(set(label_keys_train))  # 2283
    all_train_index = list(map(str, all_train_index))
    all_train_index = list(set(all_train_index).difference(temp2))

    temp_index1 = all_train_index
    train_sample_index = random.sample(temp_index1, train_num)
    temp_index1 = list(set(temp_index1).difference(set(train_sample_index)))
    query_sample_index = random.sample(temp_index1, query_num)
    temp_index1 = list(set(temp_index1).difference(set(query_sample_index)))
    database_sample_index = temp_index1

    new_train_sample, old_train_sample, new_query_sample, old_query_sample, new_database_sample, old_database_sample = [], [], [], [], [], []
    new_index_train, new_index_val = list(map(str, new_index_train)), list(map(str, new_index_val))
    for i in train_sample_index:
        if i in new_index_train:
            new_train_sample.append(i)
        else:
            old_train_sample.append(i)
    for j in query_sample_index:
        if j in new_index_train:
            new_query_sample.append(j)
        else:
            old_query_sample.append(j)
    for k in database_sample_index:
        if k in new_index_train:
            new_database_sample.append(k)
        else:
            old_database_sample.append(k)

    for id in new_train_sample:
        # print(f"train_new:{id}")
        new_train_text.append(all_text_train[id])
        new_train_image_name.append(all_image_name[id])
        l = [0] * class_number
        pointer = train_dict[id][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        new_train_label.append(vec.numpy())
    for id in old_train_sample:
        # print(f"train_old:{id}")
        old_train_text.append(all_text_train[id])
        old_train_image_name.append(all_image_name[id])
        l = [0] * class_number
        pointer = train_dict[id][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        old_train_label.append(vec.numpy())
    for id in new_query_sample:
        # print(f"query_new:{id}")
        new_query_text.append(all_text_train[id])
        new_query_image_name.append(all_image_name[id])
        l = [0] * class_number
        pointer = train_dict[id][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        new_query_label.append(vec.numpy())
    for id in old_query_sample:
        # print(f"query_old:{id}")
        old_query_text.append(all_text_train[id])
        old_query_image_name.append(all_image_name[id])
        l = [0] * class_number
        pointer = train_dict[id][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        old_query_label.append(vec.numpy())
    for id in new_database_sample:
        # print(f"database_new:{id}")
        new_database_text.append(all_text_train[id])
        new_database_image_name.append(all_image_name[id])
        l = [0] * class_number
        pointer = train_dict[id][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        new_database_label.append(vec.numpy())
    for id in old_database_sample:
        # print(f"database_old:{id}")
        old_database_text.append(all_text_train[id])
        old_database_image_name.append(all_image_name[id])
        l = [0] * class_number
        pointer = train_dict[id][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        old_database_label.append(vec.numpy())

    all_database_image_name, all_database_text, all_database_label = [], [], []
    for id in (new_database_sample + old_database_sample):
        all_database_text.append(all_text_train[id])
        all_database_image_name.append(all_image_name[id])
        l = [0] * class_number
        pointer = train_dict[id][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        all_database_label.append(vec)

    ratio = 0.2
    replay_index = random.sample(old_train_sample, math.floor(len(old_train_sample) * ratio))

    replay_image_name, replay_text, replay_label = [], [], []
    for index in replay_index:
        replay_text.append(all_text_train[index])
        replay_image_name.append(all_image_name[index])
        l = [0] * class_number
        pointer = train_dict[index][0]
        l[pointer - 1] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        replay_label.append(vec)

    indexs = ((old_train_sample, new_train_sample), (old_query_sample, new_query_sample),
              (old_database_sample, new_database_sample), replay_index, database_sample_index)
    image_names = ((old_train_image_name, new_train_image_name), (old_query_image_name, new_query_image_name),
                   (old_database_image_name, new_database_image_name), replay_image_name)
    texts = ((old_train_text, new_train_text), (old_query_text, new_query_text), (old_database_text, new_database_text),
             replay_text)
    labels = (
        (old_train_label, new_train_label), (old_query_label, new_query_label),
        (old_database_label, new_database_label), replay_label)
    return indexs, image_names, texts, labels, all_database_image_name, all_database_text, all_database_label

