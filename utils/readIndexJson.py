import json


def ReadIndexJson(data_set):
    old_samples = {}
    cluster_list = []
    if data_set == "MIRFLICKR":
        json_dir = "/home/carry6/LS_project/dataset/dataset_index/flickr_index.json"
        with open(json_dir, 'r') as f:
            clusters = json.load(f)
        count = 0
        for cluster in clusters:
            if count == 0:
                old_samples = cluster
                count = count + 1
            else:
                cluster_list.append(cluster)
    elif data_set == "COCO":
        json_dir = "/home/carry6/LS_project/dataset/dataset_index/coco_index.json"
        with open(json_dir, 'r') as f:
            clusters = json.load(f)
        count = 0
        for cluster in clusters:
            if count == 0:
                old_samples = cluster
                count = count + 1
            else:
                cluster_list.append(cluster)
    return old_samples, cluster_list
