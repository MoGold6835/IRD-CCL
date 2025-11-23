import numpy as np
import os
import random
import time
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import scipy.io as scio
from dataset.Dataset import get_dataset, TCDataset, COCODataset
from experiments.eval import retrivel
from loss.dcmh_loss import DCMH_loss
from methods.IRDCCL import IRDccl
from models import cnnf
from models import MLP
from utils.data_mining import load_mat, get_image_transform
from utils.get_alldata_coco import get_all_data
from utils.readIndexJson import ReadIndexJson


class IRDCCL():
    def __init__(self, image_dir, bit, batch_size, epoch, train_class_num, name, data_set, logger, all_num):
        """
        ECMH类
        :param image_dir: 数据集地址
        :param bit: 生成哈希码的长度
        :param batch_size: 采样大小
        :param cuda: 是否使用GPU
        """
        self.name = name
        self.data_set = data_set
        self.image_dir = image_dir
        self.bit = bit
        self.batch_size = batch_size
        self.cuda = True
        self.checkpoint_dir = os.path.join("", self.name, self.data_set,
                                           "model")
        self.save_dir = os.path.join("", self.name, self.data_set)
        self.all_num = all_num
        if data_set == "MIRFLICKR":
            self.seen_num = 22
            self.file_dir = ""
        elif data_set == "COCO":
            self.file_dir = ""

        self.train_class_num = train_class_num
        self.epoch = epoch
        self.train_num = 3
        self.num_works = 0
        self.logger = logger

        self.gamma = 0.01
        self.eta = 0.01
        self.lr = 0.000001

        self.alpha_1 = random.uniform(0.1, 0.15)
        self.beta_1 = 0.5

        self.map = {"I2T_max": 0, "T2I_max": 0, "I2T_max_old": 0, "T2I_max_old": 0, "I2T_max_all": 0, "T2I_max_all": 0,
                    "I2T": 0, "T2I": 0, "I2T_old": 0, "T2I_old": 0, "I2T_all": 0, "T2I_all": 0}
        self.best_epoch = {"best": 0, "best_old": 0, "best_all": 0}

        self.image_model = cnnf.get_cnnf(self.bit)
        self.text_model = MLP.MLP(1386, self.bit, leakRelu=False)

        self.image_optimizer = SGD(self.image_model.parameters(), self.lr)
        self.text_optimizer = SGD(self.text_model.parameters(), self.lr)
        self.optimizers = [self.image_optimizer, self.text_optimizer]

        self.old_sample, self.cluster_list = ReadIndexJson(self.data_set)

        if self.cuda:
            self.image_model = self.image_model.cuda()
            self.text_model = self.text_model.cuda()

    def train(self):

        train_sample = self.old_sample["train"]
        query_sample = self.old_sample["query"]

        transform = get_image_transform()
        if self.data_set == "MIRFLICKR":
            _, _, full_text, full_label = load_mat(self.image_dir)
            temp = "full"
            train_dataset = get_dataset(self.image_dir, full_text, full_label, transform, self.batch_size, train_sample,
                                        temp)
            eval_dataset = get_dataset(self.image_dir, full_text, full_label, transform, self.batch_size, query_sample,
                                       temp)
            # retrival_dataset = get_dataset(self.image_dir, full_text, full_label, transform, self.batch_size,
            #                                retrival_sample,temp)


        elif self.data_set == "TC12":
            temp = "full"
            dataset_path = os.path.join(self.image_dir, "IAPRTC12.mat")
            iapr_data = scio.loadmat(dataset_path)
            full_image = iapr_data['I_tr']
            full_image = full_image.astype(np.float32)
            full_text = iapr_data["T_tr"]
            full_label = iapr_data["L_tr"]
            train_dataset = TCDataset(full_image[train_sample], full_text[train_sample], full_label[train_sample],
                                      self.batch_size, transform)
            eval_dataset = TCDataset(full_image[query_sample], full_text[query_sample], full_label[query_sample],
                                     self.batch_size, transform)
        elif self.data_set == "COCO":
            temp = "full"
            full_name, full_text, full_label = get_all_data(self.image_dir)
            train_image_name, train_text, train_label = [], [], []
            query_image_name, query_text, query_label = [], [], []
            retrival_image_name, retrival_text, retrival_label = [], [], []
            for index in train_sample:
                train_image_name.append(full_name[str(index)])
                train_text.append(full_text[str(index)])
                train_label.append(full_label[str(index)])
            for index in query_sample:
                query_image_name.append(full_name[str(index)])
                query_text.append(full_text[str(index)])
                query_label.append(full_label[str(index)])
            # for index in retrival_sample:
            #     retrival_image_name.append(full_name[str(index)])
            #     retrival_text.append(full_text[str(index)])
            #     retrival_label.append(full_label[str(index)])
            train_dataset = COCODataset(self.image_dir, train_image_name, train_text, train_label, transform,
                                        self.batch_size)
            eval_dataset = COCODataset(self.image_dir, query_image_name, query_text, query_label, transform,
                                       self.batch_size)

        train_loader = DataLoader(train_dataset, self.batch_size, drop_last=True, shuffle=True,
                                  num_workers=self.num_works, pin_memory=True)
        since = time.time()
        str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
        self.image_model.train()
        self.text_model.train()
        self.logger.info(f"Starting class num is:{self.train_class_num},cluster size is:{self.train_num}")
        self.logger.info(f"{str_time} : Starting training old task samples!!!")
        self.logger.info("Original train data number:{}".format(len(self.old_sample)))
        for epoch in range(0, self.epoch):
            since1 = time.time()
            self.logger.info(f"Epoch:[{epoch + 1}/{self.epoch}]")
            loss = 0
            for data in train_loader:
                image = data["image"]
                text = data["text"]
                label = data["label"]
                if self.cuda:
                    image = image.cuda()
                    text = text.cuda()
                image_results = self.image_model(image)
                text_results = self.text_model(text)
                binary_code = torch.sign(image_results + text_results)
                sim = torch.cdist(image_results, text_results)
                loss = DCMH_loss(binary_code, image_results, text_results, sim, self.gamma, self.eta).cuda()
                loss = loss / (len(train_sample) * self.batch_size)

                self.optimizers[0].zero_grad()
                self.optimizers[1].zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                self.optimizers[0].step()
                self.optimizers[1].step()

            retrivel(eval_dataset, self.image_model, self.text_model, self.batch_size, self.cuda,
                     self.bit, self.map, self.best_epoch, epoch, self.num_works, self.all_num, flag="old")
            now = time.time() - since1
            self.logger.info("MAP_I2T:{}        MAP_T2I:{}        Loss:{}        Time:{}".format(self.map["I2T_old"],
                                                                                                 self.map["T2I_old"],
                                                                                                 loss, now))
            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    "MAX_MAP_I2T:{}         MAX_MAP_T2I:{}        BEST_EPOCH:{}".format(self.map["I2T_max_old"],
                                                                                        self.map["T2I_max_old"],
                                                                                        self.best_epoch["best_old"]))
        self.logger.info("MAX_MAP_I2T:{}         MAX_MAP_T2I:{}        BEST_EPOCH:{}".format(self.map["I2T_max_old"],
                                                                                             self.map["T2I_max_old"],
                                                                                             self.best_epoch[
                                                                                                 "best_old"]))
        self.logger.info("Old tasks are trained over.Starting training cluster classes!!!")
        IRDccl(self.cluster_list, train_sample, query_sample, self.image_model, self.text_model, self.lr,
               self.image_dir, transform, self.batch_size, self.cuda, self.bit, self.map, self.best_epoch, self.epoch,
               self.logger, self.num_works, self.data_set, self.all_num)
