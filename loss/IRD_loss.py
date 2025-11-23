import torch


def ird_loss_current(image_features, text_features, current_temp):
    # IRD_current
    features1_sim = torch.div(torch.matmul(image_features, image_features.T), current_temp)
    logits_mask = torch.scatter(
        torch.ones_like(features1_sim),
        1,
        torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
        0
    )
    logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
    features1_sim = features1_sim - logits_max1.detach()
    row_size = features1_sim.size(0)
    logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
        features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

    features2_sim = torch.div(torch.matmul(text_features, text_features.T), current_temp)
    logits_mask_2 = torch.scatter(
        torch.ones_like(features2_sim),
        1,
        torch.arange(features2_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
        0
    )
    logits_max2, _ = torch.max(features2_sim * logits_mask_2, dim=1, keepdim=True)
    features2_sim = features2_sim - logits_mask_2.detach()
    row_size_2 = features1_sim.size(0)
    logits2 = torch.exp(features2_sim[logits_mask_2.bool()].view(row_size_2, -1)) / torch.exp(
        features2_sim[logits_mask_2.bool()].view(row_size_2, -1)).sum(dim=1, keepdim=True)

    return logits1, logits2


def ird_loss_past(image_features, text_features, distill_power, past_temp, logits1_image, logits1_text):
    # IRD_past
    with torch.no_grad():
        features1_sim = torch.div(torch.matmul(image_features, image_features.T), past_temp)
        logits_mask = torch.scatter(
            torch.ones_like(features1_sim),
            1,
            torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
            0
        )
        logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
        features1_sim = features1_sim - logits_max1.detach()
        row_size = features1_sim.size(0)
        logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        features2_sim = torch.div(torch.matmul(text_features, text_features.T), past_temp)
        logits_mask_2 = torch.scatter(
            torch.ones_like(features2_sim),
            1,
            torch.arange(features2_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
            0
        )
        logits_max2, _ = torch.max(features2_sim * logits_mask_2, dim=1, keepdim=True)
        features2_sim = features2_sim - logits_max2.detach()
        logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

    loss_distill_1 = (-logits1 * torch.log(logits1_image)).sum(1).mean()
    loss_distill_2 = (-logits2 * torch.log(logits1_text)).sum(1).mean()
    return distill_power * loss_distill_1, distill_power * loss_distill_2
