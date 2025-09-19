import os
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import DETECTORS

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder

from mmdet3d.models.hbevocc.losses.semkitti import geo_scal_loss, sem_scal_loss
from mmdet3d.models.hbevocc.losses.lovasz_softmax import lovasz_softmax
from mmdet3d.models.hbevocc.atten import HeightAwareDeformableAttention
from mmdet3d.models.hbevocc.losses import HeightVoxelLoss


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@DETECTORS.register_module()
class HBEVOcc(CenterPoint):

    def __init__(self,
                 # BEVDet-Series
                 forward_projection=None,
                 # BEVFormer-Series
                 backward_projection=None,
                 # Occupancy_Head
                 occupancy_head=None,
                 # Option: Temporalfusion
                 temporal_fusion=None,
                 # Other setting
                 backward_num_layer=(1, 2),
                 class_weights=None,
                 empty_idx=17,
                 num_classes=18,
                 history_frame_num=None,
                 foreground_idx=None,
                 background_idx=None,
                 train_flow=False,
                 use_ms_feats=False,
                 save_results=False,
                 use_mask=False,
                 grid_config_bevformer=None,
                 eval_sequential=False,
                 **kwargs):
        super(HBEVOcc, self).__init__(**kwargs)
        # ---------------------- init params ------------------------------
        self.empty_idx = empty_idx
        self.backward_num_layer = backward_num_layer
        self.history_frame_num = history_frame_num
        self.foreground_idx = foreground_idx
        self.use_ms_feats = use_ms_feats
        self.save_results = save_results
        self.scene_can_bus_info = dict()
        self.grid_config_bevformer = grid_config_bevformer
        self.eval_sequential = eval_sequential
        if eval_sequential:
            print("eval_sequential: {}".format(eval_sequential))
        self.stereo_feat_prev_iv = None
        # ---------------------- init loss ------------------------------
        if self.foreground_idx is not None:
            self.flow_loss = builder.build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.use_mask = use_mask
        # ---------------------- build components ------------------------------
        # BEVDet-Series
        self.forward_projection = builder.build_neck(forward_projection)
        # BEVFormer-Series
        self._build_backward_projection(backward_projection)
        # Temporal-Fsuion
        self._build_temporal_fusion(temporal_fusion) if temporal_fusion else None
        # Simple Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head)
        self.bev_back = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                           nn.Conv2d(160, 256, kernel_size=3, stride=1, padding=1),
                                           nn.ReLU(),
                                           nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                          )
        self.back_conv = nn.Conv2d(256, 160, 1)
        self.attn_dim1 = 20 
        self.attn_head = 8
        self.points = 4
        self.self_attn_layers = 1
        self.self_attn_backward1 = nn.ModuleList(HeightAwareDeformableAttention(q_size=(100, 100), q_in_dim=160, in_dim=160, dim=self.attn_dim1, out_dim=160, n_heads=self.attn_head, points=self.points,
                              attn_drop=0.0, proj_drop=0.1, stride=1, offset_range_factor=20, use_pe=True,
                              no_off=False, ksize=3) for _ in range(self.self_attn_layers))
        self.height_voxel_loss = HeightVoxelLoss(empty_label=empty_idx, num_classes=num_classes, H=200, W=200, foreground_idx=foreground_idx, w_flow_loss=0.1)


    def _build_backward_projection(self, backward_projection_config):
        self.backward_projection = builder.build_neck(backward_projection_config)

    def _build_temporal_fusion(self, temporal_fusion_config):
        temporal_fusion_config['history_num'] = self.history_frame_num
        self.temporal_fusion = builder.build_head(temporal_fusion_config)

        x_config = self.forward_projection.img_view_transformer.grid_config['x']
        y_config = self.forward_projection.img_view_transformer.grid_config['y']
        z_config = self.forward_projection.img_view_transformer.grid_config['z']
        dx, bx, nx = gen_dx_bx(x_config, y_config, z_config)
        self.dx_for = nn.Parameter(dx, requires_grad=False)
        self.bx_for = nn.Parameter(bx, requires_grad=False)
        self.nx_for = nn.Parameter(nx, requires_grad=False)

        x_config = self.grid_config_bevformer['x']
        y_config = self.grid_config_bevformer['y']
        z_config = self.grid_config_bevformer['z']
        dx_back, bx_back, nx_back = gen_dx_bx(x_config, y_config, z_config)
        self.dx_back = nn.Parameter(dx_back, requires_grad=False)
        self.bx_back = nn.Parameter(bx_back, requires_grad=False)
        self.nx_back = nn.Parameter(nx_back, requires_grad=False)


    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def obtain_feats_from_images(self, points, img, img_metas, **kwargs):
        # 0、Prepare
        if self.with_specific_component('temporal_fusion'):
            use_temporal = True
            sequence_group_idx = torch.stack(
                [torch.tensor(img_meta['sequence_group_idx'], device=img[0].device) for img_meta in img_metas])
            start_of_sequence = torch.stack(
                [torch.tensor(img_meta['start_of_sequence'], device=img[0].device) for img_meta in img_metas])
            curr_to_prev_ego_rt = torch.stack(
                [torch.tensor(np.array(img_meta['curr_to_prev_ego_rt']), device=img[0].device) for img_meta in img_metas])
            history_fusion_params = {
                'sequence_group_idx': sequence_group_idx,
                'start_of_sequence': start_of_sequence,
                'curr_to_prev_ego_rt': curr_to_prev_ego_rt
            }
            # process can_bus info
            if 'can_bus' in img_metas[0]:
                for index, start in enumerate(start_of_sequence):
                    if start:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = 0
                        can_bus[-1] = 0
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose':temp_pose,
                            'prev_angle':temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus
                    else:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = can_bus[:3] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_pose']
                        can_bus[-1] = can_bus[-1] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_angle']
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose': temp_pose,
                            'prev_angle': temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus

        else:
            use_temporal = False
            history_fusion_params = None

        # 1、Forward-Projection create coarse voxel features
        if self.eval_sequential:
            if start_of_sequence.sum() > 0:
                self.stereo_feat_prev_iv = None
        if 'sequential' not in kwargs or not kwargs['sequential']:
            bev_feats, depth, tran_feats, ms_feats, cam_params, stereo_feat_curr_iv = self.forward_projection.extract_feat(points, img=img, img_metas=img_metas, stereo_feat_prev_iv=self.stereo_feat_prev_iv, **kwargs)
            if self.eval_sequential:
                self.stereo_feat_prev_iv = stereo_feat_curr_iv.detach().clone()
        else:
            voxel_feats, depth, tran_feats, ms_feats, cam_params = kwargs['voxel_feats'], kwargs['depth'], kwargs['tran_feats'], kwargs['ms_feats'],kwargs['cam_params']
        bs, num_cam = ms_feats[0].shape[:2]
        ms_feats[0] = self.back_conv(ms_feats[0].reshape(bs*num_cam, *ms_feats[0].shape[-3:]))

        # 2、Backward-Projection Refine
        bev_feat_back = self.backward_projection(
            mlvl_feats=ms_feats,
            img_metas=img_metas,
            cam_params=cam_params,
            pred_img_depth=None,
            # prev_bev=self.temporal_fusion.history_last_bev_back if use_temporal else None,
            # prev_bev_aug=self.temporal_fusion.history_forward_augs,
            # history_fusion_params=history_fusion_params,
        )

        # # Option: temporal fusion
        # bev_feats, bev_feat_back = self.temporal_fusion(
        #     bev_feats, bev_feat_back, cam_params, history_fusion_params, dx_for=self.dx_for, bx_for=self.bx_for, dx_back=self.dx_back, bx_back=self.bx_back,
        # )
        bev_feats = self.forward_projection.bev_encoder(bev_feats)
        bev_feat, bev_feat_down = bev_feats
        for m in self.self_attn_backward1:
            bev_feat_back = m(bev_feat_back, bev_feat_back)
        bev_feat_back_down = bev_feat_back
        bev_feat_back = self.bev_back(bev_feat_back)

        bev_feat = torch.cat([bev_feat, bev_feat_back], dim=1)
        bev_feat_down = torch.cat([bev_feat_down, bev_feat_back_down], dim=1)
        voxel_feats=[bev_feat, bev_feat_down]

        return voxel_feats, depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):

        # ---------------------- obtain feats from images -----------------------------
        voxel_feats, depth = self.obtain_feats_from_images(points, img=img_inputs, img_metas=img_metas, **kwargs)
        pred_voxel_semantic, pred_voxel_flows  = self.occupancy_head(voxel_feats)
        # ---------------------- calc loss ------------------------------
        gt_voxel_semantics = kwargs['voxel_semantics']
        if self.use_mask:
            mask_camera = kwargs['voxel_mask_camera'] # torch.Size([2, 200, 200, 16])
        else:
            mask_camera = None
        losses = dict()

        # calc forward-projection depth-loss
        loss_depth = self.forward_projection.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
        losses['loss_depth'] = loss_depth

        # calc voxel loss
        loss_occ = self.occupancy_head.loss(
            pred_voxel_semantic,  # (B, Dx, Dy, Dz, n_cls)
            gt_voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera=mask_camera
        )
        losses.update(loss_occ)
        loss_voxel = self.get_voxel_loss(
            pred_voxel_semantic,
            gt_voxel_semantics,
            mask_camera=mask_camera
        )
        losses.update(loss_voxel)
        # calc flow loss
        if self.foreground_idx is not None:
            loss_flow = self.get_flow_loss(
                    pred_voxel_flows,
                    kwargs['voxel_flows'],
                    gt_voxel_semantics,
                    loss_weight=0.1
                )
            losses.update(loss_flow)
            loss_hv = self.height_voxel_loss(preds=pred_voxel_semantic, labels=gt_voxel_semantics)
        else:
            loss_hv = self.height_voxel_loss(preds=pred_voxel_semantic, labels=gt_voxel_semantics, masks=mask_camera)
        w_hv = 0.1
        losses['loss_hv'] = w_hv * loss_hv

        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    **kwargs):
        # ---------------------- obtain feats from images -----------------------------
        voxel_feats, depth = self.obtain_feats_from_images(points, img=img_inputs[0], img_metas=img_metas, **kwargs)

        # ---------------------- forward ------------------------------
        pred_voxel_semantic, pred_voxel_flows  = self.occupancy_head(voxel_feats)
        return_dict = dict()
        pred_voxel_semantic_cls = pred_voxel_semantic.softmax(-1).argmax(-1)
        if self.foreground_idx is not None:
            foreground_mask = torch.zeros(pred_voxel_semantic_cls.shape).to(pred_voxel_semantic_cls.device)
            for idx in self.foreground_idx:
                foreground_mask[pred_voxel_semantic_cls == idx] = 1
            return_pred_voxel_flows = torch.zeros_like(pred_voxel_flows)
            return_pred_voxel_flows[foreground_mask != 0] = pred_voxel_flows[foreground_mask != 0]
            return_dict['flow_results'] = return_pred_voxel_flows.cpu().numpy().astype(np.float16)
        
        return_dict['occ_results'] = pred_voxel_semantic_cls.cpu().numpy().astype(np.uint8)
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        if self.save_results:
            sample_idx = [img_meta['sample_idx'] for img_meta in img_metas]
            scene_name = [img_meta['scene_name'] for img_meta in img_metas]
            # check save_dir
            for name in scene_name:
                if not os.path.exists('results/{}'.format(name)):
                    os.makedirs('results/{}'.format(name))
            for i, idx in enumerate(sample_idx):
                np.savez('results/{}/{}.npz'.format(scene_name[i], idx),semantics=return_dict['occ_results'][i], flow=return_dict['flow_results'][i])
        return [return_dict]

    def get_voxel_loss(self,
                    pred_voxel_semantic,
                    target_voxel_semantic,
                    mask_camera=None
                    ):
        if mask_camera is not None:
            target_voxel_semantic[torch.where(mask_camera==0)] = 255
        # change pred_voxel_semantic from [bs, w, h, z, c] -> [bs, c, w, h, z]  !!!
        pred_voxel_semantic = pred_voxel_semantic.permute(0, 4, 1, 2, 3)
        # print("pred_voxel_semantic.shape:{}".format(pred_voxel_semantic.shape)) # torch.Size([2, 17, 200, 200, 16]) torch.Size([2, 17, 100, 100, 1])
        loss_dict = {}

        w_lovasz = 1 # 0.5
        w_sem_scal = 0.1
        w_geo_scal = 0.1
        loss_dict['loss_lovasz'] = w_lovasz * lovasz_softmax(
            torch.softmax(pred_voxel_semantic, dim=1),
            target_voxel_semantic,
            ignore=255,
        )
        loss_dict['loss_sem_scal'] = w_sem_scal * sem_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255)
        loss_dict['loss_geo_scal'] = w_geo_scal * geo_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255,
            empty_idx=self.empty_idx)

        return loss_dict

    def get_flow_loss(self, pred_flows, target_flows, target_sem, loss_weight):
        loss_dict = {}

        loss_flow = 0
        for i in range(target_flows.shape[0]):
            foreground_mask = torch.zeros(target_flows[i].shape[:-1])
            for idx in self.foreground_idx:
                foreground_mask[target_sem[i] == idx] = 1

            pred_flow = pred_flows[i][foreground_mask!=0]
            target_flow = target_flows[i][foreground_mask!=0]

            loss_flow = loss_flow + loss_weight * self.flow_loss(pred_flow, target_flow)
        loss_dict['loss_flow'] = loss_flow

        return loss_dict

    