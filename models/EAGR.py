import torch
from torch import nn
import torch.nn.functional as F

from .backbones import BACKBONES


class EdgeModule(nn.Module):
    def __init__(self, in_fea=(256, 512, 1024), mid_fea=256, out_fea=2):
        super(EdgeModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.shape

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv, nn.BatchNorm2d(out_features), nn.ReLU())

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        for stage in self.stages:
            x = stage(feats)

        priors = [F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class DecoderModule(nn.Module):
    def __init__(self, in_plane1, in_plane2, num_classes):
        super(DecoderModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_plane2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.shape

        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class EAGRModule(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):
        super(EAGRModule, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = nn.Sequential(nn.BatchNorm2d(num_in), nn.ReLU())

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        # Construct projection matrix
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        # Project and graph reason
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        # Reproject
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))

        return out


class EAGRNet(nn.Module):
    def __init__(self, num_classes, backbone="resnet101"):
        super().__init__()

        self.backbone = BACKBONES[backbone](pretrained=True, dilation=(False, True, True), return_features=True)

        self.layer5 = PSPModule(2048, 512)
        self.edge_layer = EdgeModule()
        self.block1 = EAGRModule(512, 128, 4)
        self.block2 = EAGRModule(256, 64, 4)
        self.layer6 = DecoderModule(512, 256, num_classes)

    def forward(self, x):
        N, C, H, W = x.shape

        x1, x2, x3, x4, x5 = self.backbone(x)

        x = self.layer5(x5)
        edge, edge_fea = self.edge_layer(x2, x3, x4)
        x = self.block1(x, edge.detach())
        x2 = self.block2(x2, edge.detach())
        seg, x = self.layer6(x, x2)

        seg = F.upsample(seg, size=(H, W), mode='bilinear')
        edge = F.upsample(edge, size=(H, W), mode='bilinear')
        return seg, edge
