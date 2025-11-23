import torch
from sklearn.metrics import precision_recall_curve

from loss.tri_loss import TripletLoss


def calc_map_k(query_data, retrieval_data, query_label, retrieval_label, k=None):
    # qB:查询集  范围{-1,+1}
    # rB:检索集  范围{-1,+1}
    # query_label: 查询标签
    # retrieval_label: 检索标签
    num_query = query_label.size(0)  # 查询个数
    map = 0.
    if k is None:
        k = retrieval_label.size(0)  # 如果不指定k,k将是全部检索个数。对于flickr25k数据集，即18015
    for iter in range(num_query):
        # 每个查询标签乘以检索标签的转置，只要有相同标签，该位置就是1
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.transpose(0, 1)) > 0).type(
            torch.float).squeeze()
        tsum = torch.sum(gnd)  # 真实相关的数据个数
        # print("相关个数：", tsum)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(query_data[iter, :], retrieval_data)
        _, ind = torch.sort(hamm)  # ind ：已排序的汉明距，在未排序中的位置
        ind.squeeze_()
        # print("原始 gnd:", gnd)
        # print("ind    :", ind)
        gnd = gnd[ind]  # 按照预测的顺序重排
        # print("重排后gnd:", gnd)
        total = min(k, int(tsum))  # 取k和tsum的最小值，这句应该没啥用
        # 如果有三个相关的，则count是[1，2，3]
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        # 取出重排后非0元素的位置
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        # print("count:", count)
        # print("tindex:", tindex)
        map += torch.mean(count / tindex)
        # print("map:", map)
    map = map / num_query
    return map


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2,
              alpha, beta, gamma):
    tri_loss = TripletLoss(reduction="mean")
    tri_i2t = tri_loss.forward(view1_feature, labels_1.float(), target=view2_feature, margin=0.1)
    tri_t2i = tri_loss.forward(view2_feature, labels_2.float(), target=view1_feature, margin=0.1)
    cos_tri = tri_i2t + tri_t2i

    floss = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()
    lloss = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
            (view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

    im_loss = alpha * cos_tri + beta * floss + gamma * lloss

    return im_loss, cos_tri, floss, lloss


def calc_hamming_dist(B1, B2):
    q = B2.size(1)  # 哈希码位数
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))  # 计算汉明码距离 公式4
    return distH


def calc_neighbor(label1, label2):
    """
    计算模态数据之间的相似矩阵
    :param label1: 模态1的标签
    :param label2: 模态2的标签
    :return: 相似矩阵
    """
    label1 = label1.to(torch.float)
    label2 = label2.to(torch.float)
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    # 枚举 query sample
    # precision_recall_curve()
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # 整个被检索数据库中的相关样本数
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R
