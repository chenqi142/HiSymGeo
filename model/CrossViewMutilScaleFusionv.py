import torch
import torch.nn as nn
import torch.nn.functional as F

from .DynamicFeatureSelector import DynamicFeatureSelector

class CrossViewMutilScaleFusion(nn.Module):
    def __init__(self):
        super(CrossViewMutilScaleFusion, self).__init__()
        self.moe = DynamicFeatureSelector(512, 3)
    
    def forward(self, global_query, values, training=True):
        context1, attn1 = self._fusion(global_query, values[0])
        context2, attn2 = self._fusion(global_query, values[1])
        context3, attn3 = self._fusion(global_query, values[2])

        # 获得最大分辨率
        max_h = max(context1.shape[2], context2.shape[2], context3.shape[2])
        max_w = max(context1.shape[3], context2.shape[3], context3.shape[3])

        # 上采样到最大分辨率
        context1_upsampled = F.interpolate(context1, size=(max_h, max_w), mode='bilinear', align_corners=False)
        context2_upsampled = F.interpolate(context2, size=(max_h, max_w), mode='bilinear', align_corners=False)
        context3_upsampled = F.interpolate(context3, size=(max_h, max_w), mode='bilinear', align_corners=False)

        attn1_upsampled = F.interpolate(attn1, size=(max_h, max_w), mode='bilinear', align_corners=False)
        attn2_upsampled = F.interpolate(attn2, size=(max_h, max_w), mode='bilinear', align_corners=False)
        attn3_upsampled = F.interpolate(attn3, size=(max_h, max_w), mode='bilinear', align_corners=False)

        # 融合
        fused_context = torch.max(torch.stack([context1_upsampled, context2_upsampled, context3_upsampled], dim=0), dim=0)[0]
        fused_attn = torch.max(torch.stack([attn1_upsampled, attn2_upsampled, attn3_upsampled], dim=0), dim=0)[0]

        fused_features = self.moe(global_query, [fused_context, context2_upsampled, context3_upsampled], training)

        return fused_features, fused_attn
    
    def _fusion(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W*H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1,2))
        score = score.view(B, W*H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)
        
        attn = torch.zeros(B, H*W).to(value.device)
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])
        
        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn