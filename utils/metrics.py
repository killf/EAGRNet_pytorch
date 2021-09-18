import torch


@torch.no_grad()
def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes=1):
    ious = []
    ious_sum = 0
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(1, num_classes):
        pred_indices = pred == cls
        target_indices = target == cls

        intersection = (pred_indices[target_indices]).long().sum().item()
        union = pred_indices.long().sum().item() + target_indices.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            iou = float(intersection) / max(float(union), 1)
            ious.append(iou)
            ious_sum += iou
    return ious_sum / num_classes
