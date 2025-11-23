import random

from experiments.fixedIRDCCL import IRDCCL


def get_trainer(image_dir, batch_size, bit, epoch, name, data_set, logger, all_num):
    if name == "IRDCCL":
        train_class_num = 4
        return IRDCCL(image_dir, bit, batch_size, epoch, train_class_num, name, data_set, logger, all_num)
    else:
        logger.info("Error:no method match!!!")
        pass
