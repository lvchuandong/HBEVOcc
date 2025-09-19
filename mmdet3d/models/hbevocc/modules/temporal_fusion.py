import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32

from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class TemporalFusion(BaseModule):
    def __init__(
            self,
            top_k=None,
            history_num=8,
            single_bev_back_num_channels=None,
            single_bev_for_num_channels=None,
            foreground_idx=None,
            num_classes=17,
            **kwargs
    ):
        super(TemporalFusion, self).__init__()
        self.single_bev_back_num_channels = single_bev_back_num_channels
        self.single_bev_for_num_channels = single_bev_for_num_channels
        
        self.history_bev_for = None
        self.history_bev_back = None
        self.history_last_bev_for = None
        self.history_last_bev_back = None
        
        self.history_forward_augs = None
        self.history_num = history_num
        self.history_seq_ids = None
        self.history_sweep_time = None
        self.history_cam_sweep_freq = 0.5       # seconds between each frame
        self.top_k = top_k                      # top_k sampling
        self.foreground_idx = foreground_idx    # Set the foreground index

        self.conv = nn.Sequential(nn.Conv2d(320, 160, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU())

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev

    @force_fp32()
    def forward(self, curr_bev_for, curr_bev_back, cam_params, history_fusion_params, dx_for, bx_for, dx_back, bx_back,
                history_last_bev_for=None, history_last_bev_back=None, 
                last_occ_pred=None, nonempty_prob=None):

        if len(curr_bev_for.shape) == 4:
            curr_bev_for = curr_bev_for.unsqueeze(2)
            curr_bev_back = curr_bev_back.unsqueeze(2)

        # 1、Get some history fusion information
        # Process test situation
        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()

        # 2、Deal with first batch
        if self.history_bev_back is None:
            self.history_bev_for = curr_bev_for.clone()
            self.history_bev_for = curr_bev_for.repeat(1, self.history_num, 1, 1, 1)
            self.history_bev_back = curr_bev_back.clone()
            self.history_bev_back = curr_bev_back.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev_for.new_zeros(curr_bev_for.shape[0], self.history_num)
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()

        assert self.history_bev_for.dtype == torch.float32
        assert self.history_bev_back.dtype == torch.float32

        # 3、 Deal with the new sequences
        # Replace all the new sequences' positions in history with the curr_bev information
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        self.history_sweep_time += 1
        
        if start_of_sequence.sum() > 0:
            self.history_bev_for[start_of_sequence] = curr_bev_for[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            self.history_bev_back[start_of_sequence] = curr_bev_back[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time[start_of_sequence] = 0
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]

        # backward fusion
        tmp_bev_back = self.history_bev_back
        bs, mc, z, h, w = tmp_bev_back.shape
        n, c_, z, h, w = curr_bev_back.shape
        grid_back = self.generate_grid(curr_bev_back)
        feat2bev_back = self.generate_feat2bev(grid_back, dx_back, bx_back)
        rt_flow_back = (torch.inverse(feat2bev_back) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev_back)
        grid_back = rt_flow_back.view(n, 1, 1, 1, 4, 4) @ grid_back
        normalize_factor_back = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev_back.dtype, device=curr_bev_back.device)
        # print("normalize_factor:{}".format(normalize_factor)) # normalize_factor:tensor([99., 99.,  0.], device='cuda:0')
        grid_back = grid_back[:, :, :, :, :3, 0] / normalize_factor_back.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z
        tmp_bev_back = tmp_bev_back.reshape(bs, mc, z, h, w)
        sampled_history_bev_back = F.grid_sample(tmp_bev_back, grid_back.to(curr_bev_back.dtype).permute(0, 3, 1, 2, 4),  
                                               align_corners=True, mode='bilinear')
        sampled_history_bev_back = sampled_history_bev_back.reshape(n, mc, z, h, w)
        curr_bev_back = curr_bev_back.reshape(n, c_, z, h, w)
        feats_cat_back = torch.cat([curr_bev_back, sampled_history_bev_back], dim=1)
        curr_bev_back = self.conv(feats_cat_back.squeeze(2))
        feats_to_return_back = curr_bev_back

        # forward fusion
        tmp_bev_for = self.history_bev_for
        bs, mc, z, h, w = tmp_bev_for.shape
        n, c_, z, h, w = curr_bev_for.shape
        grid_for = self.generate_grid(curr_bev_for)
        feat2bev_for = self.generate_feat2bev(grid_for, dx_for, bx_for)
        rt_flow_for = (torch.inverse(feat2bev_for) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev_for)
        grid_for = rt_flow_for.view(n, 1, 1, 1, 4, 4) @ grid_for
        normalize_factor_for = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev_for.dtype, device=curr_bev_for.device)
        grid_for = grid_for[:, :, :, :, :3, 0] / normalize_factor_for.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z
        tmp_bev_for = tmp_bev_for.reshape(bs, mc, z, h, w)
        sampled_history_bev_for = F.grid_sample(tmp_bev_for, grid_for.to(curr_bev_for.dtype).permute(0, 3, 1, 2, 4),  
                                               align_corners=True, mode='bilinear')
        curr_bev_for = curr_bev_for.reshape(n, c_, z, h, w)
        feats_cat_for = torch.cat([curr_bev_for, sampled_history_bev_for], dim=1)
        feats_to_return_for = feats_cat_for.squeeze(2)

        # update history information
        self.history_bev_for = feats_cat_for[:, :-self.single_bev_for_num_channels, ...].detach().clone()
        self.history_bev_back = feats_cat_back[:, :-self.single_bev_back_num_channels, ...].detach().clone()
        self.history_last_bev_back = feats_to_return_back.unsqueeze(2).detach().clone()
        self.history_forward_augs = forward_augs.clone()

        return feats_to_return_for.clone(), feats_to_return_back.clone()