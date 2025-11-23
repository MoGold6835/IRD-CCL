import random
import torch
from torchvision import transforms


def synonym_replace(text_vectors):
    # 实现同义词替换的函数
    # 可以使用词典或同义词库来进行替换操作
    # 返回替换后的文本向量数组
    replaced_vectors = [...]
    return replaced_vectors


def random_insertion(text_vectors, n):
    # 实现随机插入的函数
    # 在文本向量数组中随机插入n个额外的单词、短语或句子
    # 返回插入后的文本向量数组
    inserted_vectors = [...]
    return inserted_vectors


def random_deletion(text_vectors, p):
    # 实现随机删除的函数
    # 随机删除文本向量数组中的一些单词或短语
    # 根据概率p决定是否删除每个单词
    # 返回删除后的文本向量数组
    deleted_vectors = [...]
    return deleted_vectors


def random_swap(text_vectors, n):
    # 实现随机交换的函数
    # 随机交换文本向量数组中的n对单词或短语
    # 返回交换后的文本向量数组
    swapped_vectors = [...]
    return swapped_vectors


def random_replace(text_tensor, vocab_size):
    # 随机选择要替换的位置
    replace_index = random.randint(0, len(text_tensor) - 1)

    # 随机选择替换的词
    replacement = random.randint(0, vocab_size - 1)

    # 替换文本中的词
    # text_tensor[replace_index] = replacement
    if text_tensor[replace_index] == 1:
        text_tensor[replace_index] = 0
    else:
        text_tensor[replace_index] = 1
    return text_tensor


def clone_text(text):
    temp_text = torch.clone(text)
    return temp_text


def clone_label(label):
    temp_label = torch.clone(label)
    return temp_label


def augmentation(data):
    image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(45),  # 随机旋转 (范围：-45度到+45度)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色调整
    ])
    images = data["image"]
    texts = data["text"]
    labels = data["label"]
    augmented_images, augmented_texts, augmented_labels = [], [], []
    for image in images:
        temp_image = image_transform(image)
        temp_image = temp_image.unsqueeze(0)
        augmented_images.append(temp_image)
    for text in texts:
        augmented_vector = clone_text(text)
        augmented_vector = augmented_vector.unsqueeze(0)
        # print(augmented_vector.type())
        augmented_texts.append(augmented_vector)
    for label in labels:
        augmented_label = clone_label(label)
        augmented_label=augmented_label.unsqueeze(0)
        augmented_labels.append(augmented_label)
    augmented_labels = torch.cat(augmented_labels, dim=00)
    augmented_images = torch.cat(augmented_images, dim=0)
    augmented_texts = torch.cat(augmented_texts, dim=0)
    return augmented_images, augmented_texts, augmented_labels
