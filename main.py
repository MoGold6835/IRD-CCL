import logging
import os
import time
from utils.trainer import get_trainer


def init_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main():
    trainer = get_trainer(image_dir, batch_size, bit, epoch, name, data_set, logger, all_num)
    since = time.time()
    str2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
    logger.info(f"{str2} ##Method Name:{name}  ##Dataset Name:{data_set}")
    logger.info("Start training old tasks!!!")
    logger.info("Image neural network：CNN-F   Text neural network：MLP")
    logger.info("Start model training......")
    trainer.train()
    logger.info("*" * 50)


if __name__ == "__main__":
    base_dir = os.path.join(project_dir, "dataset")
    name = "IRD-CCL"
    data_set = "MIRFLICKR"
    if data_set == "MIRFLICKR":
        image_dir = os.path.join(base_dir, "mirflickr")
        all_num = 24
    elif data_set == "COCO":
        image_dir = os.path.join(base_dir, "coco")
        all_num = 80

    batch_size = 64
    bit = 64
    epoch = 50

    log_file = os.path.join(project_dir, "IRD-CCL/checkpoints", name, data_set, "logging.log")
    logger = init_logging(log_file)
    main()
