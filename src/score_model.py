import torch
import torch.nn as nn
import numpy as np

from .gnn_feature import PolygonGCN

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PolygonPackingTransformer(nn.Module):
    def __init__(self, marginal_prob_std_func, device, feature_dim=64, hidden_dim=128, nhead=16, maxInputLength=50, num_encoder_layers=8, num_decoder_layers=8):
        super(PolygonPackingTransformer, self).__init__()
        
        self.marginal_prob_std = marginal_prob_std_func
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.maxInputLength = maxInputLength
        
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(True)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
        )

        self.geo_feature = PolygonGCN(self.feature_dim)
        
        self.transformer_model = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, 4)  # 输出层，将Transformer的输出映射到2个值，即x和y速度

    def forward(self, actions, polyIds, paddingMask, batch, t, gnnFeatureData, polyFeatures=None):
        batchCounts = torch.bincount(batch)
        batchSize = int(batch.max() + 1)
        
        sigma_feature = self.t_embed(t.unsqueeze(-1))
        actions_feature = self.action_encoder(actions).reshape(batchSize, -1, self.feature_dim)
        
        if polyFeatures is None:
            geo_feature_all = self.geo_feature(gnnFeatureData) # polyCnt, e * fd
        else:
            geo_feature_all = polyFeatures

        polyIds = polyIds.unsqueeze(1).expand(-1, self.feature_dim) 
        # print(polyIds.shape, featureA.shape)
        geo_feature = torch.gather(geo_feature_all, 0, polyIds).reshape(batchSize, -1, self.feature_dim)

        semantic_geo_pos = torch.cat([geo_feature, actions_feature], dim=-1) # batch, maxCnt, fd * 2
        combined_feature_input = torch.cat([semantic_geo_pos, sigma_feature], dim=1) # batch, maxCnt + 1, fd * 2
        # paddingMask is a float tensor, where -inf means masked
        # paddingMask is used for conbine feature input, but with shape batch * maxCnt, 1
        paddingMask = paddingMask.reshape(batchSize, -1)
        paddingMaskInput = torch.cat([paddingMask, torch.zeros(batchSize, 1).to(self.device)], dim=1)

        transformer_out = self.transformer_model(
            combined_feature_input, 
            combined_feature_input,
            src_key_padding_mask=paddingMaskInput,
            tgt_key_padding_mask=paddingMaskInput)
        # 计算输出
        
        mu, std = self.marginal_prob_std(torch.zeros_like(t), t)
        std = torch.repeat_interleave(std, batchCounts).view(-1, 1) + 1e-5
        
        velocities = self.output_layer(transformer_out)[:, :-1].reshape(-1, 4)
        # print(velocities.shape)
        return velocities / std
