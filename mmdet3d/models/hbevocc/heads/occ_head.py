import copy
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import reduce_mean
from mmdet.models import HEADS

from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.hbevocc.modules.basic_block import BasicBlock3D
from mmcv.cnn import ConvModule
from mmdet3d.models.builder import build_loss

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_classes=17,
            foreground_idx=None,
            bev_w=200,
            bev_h=200,
            bev_z=16,
            loss_occ=None,
    ):
        super(OccHead, self).__init__()
        self.final_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.foreground_idx = foreground_idx
        self.up = nn.Sequential(nn.Conv2d(160+160, in_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                )
        self.predicter = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
                nn.Softplus(),
                nn.Conv2d(out_channels, num_classes*16, kernel_size=1, padding=0),
            )
        if self.foreground_idx is not None:
            self.predicter_flow = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
                    nn.Softplus(),
                    nn.Conv2d(out_channels, 2*16, kernel_size=1, padding=0),
                )

        self.foreground_idx = foreground_idx
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.bev_z = bev_z
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    @force_fp32()
    def forward(self, img_feats):
        img_feats, img_feats_down = img_feats
        img_feats_down = self.up(img_feats_down)
        img_feats = img_feats + img_feats_down
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_feats = self.final_conv(img_feats)
        bs, C, Dy, Dx = occ_feats.shape

        # occ
        occ_pred = self.predicter(occ_feats)
        occ_pred = occ_pred.permute(0, 3, 2, 1)
        occ_pred = occ_pred.view(bs, Dx, Dy, self.bev_z, self.num_classes)
        # flow
        if self.foreground_idx is not None:
            occ_flow = self.predicter_flow(occ_feats)
            occ_flow = occ_flow.permute(0, 3, 2, 1)
            occ_flow = occ_flow.view(bs, Dx, Dy, self.bev_z, 2)
        else:
            occ_flow = None

        return occ_pred, occ_flow
    
    def loss(self, occ_pred, voxel_semantics, mask_camera=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if mask_camera is not None:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_occ'] = loss_occ
        else:
            occ_voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, occ_voxel_semantics)
            loss['loss_occ'] = loss_occ
        return loss