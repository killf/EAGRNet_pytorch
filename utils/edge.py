import torch


def generate_edge(mask):
    """
    Calculate the ege for the mask.
    :param mask: B,H,W
    :return: B,H,W
    """
    left = torch.roll(mask, 1, 2)
    right = torch.roll(mask, -1, 2)
    top = torch.roll(mask, 1, 1)
    bottom = torch.roll(mask, -1, 1)

    a = torch.not_equal(mask, left)
    b = torch.not_equal(mask, right)
    c = torch.not_equal(mask, top)
    d = torch.not_equal(mask, bottom)

    t1 = torch.logical_or(a, b)
    t2 = torch.logical_or(c, d)
    edge = torch.logical_or(t1, t2)
    return edge.type(torch.int64)
