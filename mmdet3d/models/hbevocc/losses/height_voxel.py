import torch
import torch.nn as nn
import torch.nn.functional as F

class HeightVoxelLoss(nn.Module):
    def __init__(self, empty_label=16, num_classes=17, H=200, W=200, foreground_idx=None):
        super(HeightVoxelLoss, self).__init__()
        self.criterion = nn.SmoothL1Loss()
        self.empty_label = empty_label
        self.num_classes = num_classes
        self.choose_num = 4000
        self.all_indices = torch.stack(torch.meshgrid(
            torch.arange(H), 
            torch.arange(W),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)
        self.min_weight = 1.0  # 最小权重
        self.max_weight = 3.0 # 3.0  # 最大权重
        self.height = 16
        self.foreground_idx = foreground_idx

    def forward(self, preds, labels, masks=None):
        device = preds.device
        B, X, Y, Z, C = preds.shape
        labels = labels.type(torch.LongTensor).to(device)
        loss = 0
        # loss_num = 0
        h = Z // self.height
        
        for bs in range(B):
            pred_fbg = preds[bs]
            label_fbg = labels[bs]
            selected_indices = self.all_indices[torch.randperm(X * Y)[:self.choose_num]]
            x_coords = selected_indices[:, 0]
            y_coords = selected_indices[:, 1]
            pred_fbg = pred_fbg[x_coords, y_coords, :, :]
            label_fbg = label_fbg[x_coords, y_coords, :]
            prob_bs = torch.empty(0).type_as(pred_fbg)
            prob_label_bs = torch.empty(0).type_as(label_fbg)
            
            if masks is not None:
                mask_fbg = masks[bs]
                mask_fbg = mask_fbg[x_coords, y_coords, :].type_as(preds)
            
            # 先统计每个高度层的有效样本数
            height_counts = torch.zeros(self.height, device=device)
            for i in range(self.height):
                pred = pred_fbg[:, i*h:(i+1)*h, :]
                label = label_fbg[:, i*h:(i+1)*h]
                pred = pred.reshape(self.choose_num*h, C)
                label = label.reshape(-1)
                
                if masks is not None:
                    mask = mask_fbg[:, i*h:(i+1)*h]
                    mask = mask.reshape(-1)
                    use_where = torch.where((label!=self.empty_label)&(mask!=0))
                else:
                    use_where = torch.where(label!=self.empty_label)
                
                if len(use_where[0])>0:
                    height_counts[i] = len(use_where[0])

            # 计算每个高度层的权重（与样本数成反比）
            max_count = height_counts.max()
            ratio = self.min_weight / self.max_weight
            weights = torch.where(
                height_counts > 0,
                self.max_weight * (ratio ** (height_counts / max_count)),
                torch.tensor(0.0, device=device)
            )
            
            # 计算损失
            for i in range(self.height):
                pred = pred_fbg[:, i*h:(i+1)*h, :]
                label = label_fbg[:, i*h:(i+1)*h]
                pred = pred.reshape(self.choose_num*h, C)
                label = label.reshape(-1)
                
                if masks is not None:
                    mask = mask_fbg[:, i*h:(i+1)*h]
                    mask = mask.reshape(-1)
                    use_where = torch.where((label!=self.empty_label)&(mask!=0))
                else:
                    use_where = torch.where(label!=self.empty_label)
                
                if len(use_where[0])>0:
                    pred = pred[use_where]
                    label = label[use_where]
                    pred_max = torch.max(pred,dim=-1,keepdim=True)[0]
                    pred = pred - pred_max
                    pred_prob = F.softmax(pred, dim=-1)
                    prob = pred_prob[torch.arange(len(use_where[0])), label]
                    prob_label = torch.zeros_like(prob).type_as(preds)
                    
                    # 应用动态权重
                    weighted_prob = weights[i] * torch.log(prob+1e-3)
                    prob_bs = torch.cat([prob_bs, weighted_prob])
                    prob_label_bs = torch.cat([prob_label_bs, prob_label])
            
            loss += self.criterion(prob_bs, prob_label_bs)

        loss = loss / B
        return loss