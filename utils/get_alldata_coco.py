import json
import os

import torch

base_dir = "/home/carry6/LS_project/dataset/coco"


def get_all_data(base_dir):
    annotation_dir = os.path.join(base_dir, "annotations_trainval2014", "annotations")
    coco_preprocess_data = os.path.join(annotation_dir, "coco_preprocess_data.json")
    coco_vector = os.path.join(annotation_dir, "coco_textvector_data.json")
    json_dir = os.path.join(annotation_dir, "coco_label_data.json")

    with open(os.path.join(coco_preprocess_data), "r", encoding="utf-8") as f:
        data = f.read()
    json_data = json.loads(data)
    f.close()

    train_data = json_data["train_data"]

    with open(os.path.join(coco_vector), "r", encoding="utf-8") as f:
        data = f.read()
    vector_data = json.loads(data)
    f.close()

    with open(json_dir, "r", encoding="utf-8") as f:
        data = f.read()
    label_data = json.loads(data)
    f.close()

    all_name = train_data[0]
    all_text = vector_data["train"]

    all_label = {}
    for key, value in label_data["train"].items():
        l = [0] * 80
        for i in value:
            pointer = i - 1
            l[pointer] = 1
        vec = torch.tensor(l, dtype=torch.float32)
        all_label[key] = vec

    return all_name, all_text, all_label

