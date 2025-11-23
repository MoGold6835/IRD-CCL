import torch


def SupConLoss(features, labels, temperature=0.07, base_temperature=0.07, reduction='mean'):
    """

    :param features: 样本特征
    :param labels: 样本标签
    :param net: 当前样本的索引
    :param target_labels:目标标签，掩膜
    :param temperature: 温度参数
    :param base_temperature: 基础温度参数
    :param contrast_mode: 对比学习的模式
    :param reduction:
    :return:
    """
    loss = 0
    for index in range(features.size(0)):
        positive_points = []
        negative_points = []
        for label_index in range(labels.size(0)):
            if label_index == index:
                continue
            else:
                if labels[index].equal(labels[label_index]):
                    positive_points.append(label_index)
                else:
                    negative_points.append(label_index)
        num_positive = len(positive_points)
        paramiter = -1 / abs(num_positive)
        temp_loss = 0
        for positive_index in positive_points:
            anchor_feature = features[index]
            postive_features = features[positive_index]
            logit1 = torch.matmul(anchor_feature, postive_features.T) / (torch.tensor(temperature))
            logit1 = torch.exp(logit1)
            temp_logits = 0
            for negative_index in negative_points:
                negative_feature = features[negative_index]
                logit2 = torch.matmul(anchor_feature, negative_feature) / (torch.tensor(base_temperature))
                logit2 = torch.exp(logit2)
                temp_logits = temp_logits + logit2
            ratio = logit1 / temp_logits
            log_ratio = torch.log(ratio)
            temp_loss = temp_loss + log_ratio
        loss = loss + temp_loss * paramiter
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'none':
        loss = loss.mean(0)
    else:
        raise ValueError('loss reduction not supported: {}'.
                         format(reduction))
    return loss
