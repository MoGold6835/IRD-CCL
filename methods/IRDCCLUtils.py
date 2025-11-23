import torch
from torch.utils.data import DataLoader
from utils.calculate import calc_map_k


def IRDCCLRetrival(query_data, image_model, text_model, batch_size, cuda, bit, map, best_epoch, epoch, num_works,
                   all_num, flag):
    query_dataloader = DataLoader(query_data, batch_size, shuffle=True, pin_memory=True, drop_last=True,
                                  num_workers=num_works)

    image_hash_buffer, text_hash_buffer, label_buffer = torch.zeros(len(query_data), bit), \
                                                        torch.zeros(len(query_data), bit), \
                                                        torch.zeros(len(query_data), all_num)
    if cuda:
        image_hash_buffer = image_hash_buffer.cuda()
        text_hash_buffer = text_hash_buffer.cuda()
        label_buffer = label_buffer.cuda()
    image_model.eval()
    text_model.eval()
    with torch.no_grad():
        for data in query_dataloader:
            image = data["image"]
            text = data["text"]
            label = data["label"]
            index = data["index"].numpy()
            if cuda:
                image = image.cuda()
                text = text.cuda()
                label = label.cuda()
            image_results = image_model(image)
            text_results = text_model(text)
            image_hash = torch.sign(image_results)
            text_hash = torch.sign(text_results)
            image_hash_buffer[index, :] = image_hash
            text_hash_buffer[index, :] = text_hash
            label_buffer[index, :] = label

    image2text = calc_map_k(image_hash_buffer, text_hash_buffer, label_buffer, label_buffer).cuda()
    text2image = calc_map_k(text_hash_buffer, image_hash_buffer, label_buffer, label_buffer).cuda()

    if flag == "old":
        map["I2T_old"] = image2text
        map["T2I_old"] = text2image
        if image2text > map["I2T_max_old"]:
            map["I2T_max_old"] = image2text
        if text2image > map["T2I_max_old"]:
            map["T2I_max_old"] = text2image
            best_epoch["best_old"] = epoch + 1
    elif flag == "new":
        map["I2T"] = image2text
        map["T2I"] = text2image
        if image2text > map["I2T_max"]:
            map["I2T_max"] = image2text
        if text2image > map["T2I_max"]:
            map["T2I_max"] = text2image
            best_epoch["best"] = epoch + 1
    elif flag == "all":
        map["I2T_all"] = image2text
        map["T2I_all"] = text2image
        if image2text > map["I2T_max_all"]:
            map["I2T_max_all"] = image2text
        if text2image > map["T2I_max_all"]:
            map["T2I_max_all"] = text2image
            best_epoch["best_all"] = epoch + 1
