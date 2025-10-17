import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, x, key_padding_mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        # Reshape x to [seq_len, batch_size, input_dim]
        x = x.transpose(0, 1)
        
        batch_size = x.size(1)
        query = self.query.repeat(1, batch_size, 1)
        attn_output, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        # attn_output shape: [1, batch_size, output_dim]
        
        # Reshape output to [batch_size, output_dim]
        return attn_output.squeeze(0) # 


class PolygonGCN(nn.Module):
    def __init__(self, out_feature):
        super(PolygonGCN, self).__init__()
        self.initLin = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(True)
        )
        
        globalEncNum = 8
        d_model = 64
        
        self.conv1 = GCNConv(16, 16)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(48, 16)
        self.conv4 = GCNConv(64, d_model - globalEncNum)
        self.global_fc = nn.Linear(1, globalEncNum) # in feature is 1, out feature is 16
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
        
        self.att_pool = AttentionPooling(d_model, 8)
        
        
    def paddingToEachBatch(self, x, batch):
        inf = 1e9
        batch_size = batch.max() + 1
        max_batch_size = batch.bincount().max()

        range_tensor = torch.arange(max_batch_size, device=x.device).expand(batch_size, -1)
        batch_counts = batch.bincount().unsqueeze(-1).expand_as(range_tensor)

        mask = range_tensor < batch_counts

        batched_data = mask.unsqueeze(-1).expand(-1, -1, x.size(-1)).float()
        batched_data[mask] = x
        
        paddingMask = torch.ones_like(mask).float() * -inf
        paddingMask[mask] = 0

        return batched_data, paddingMask
        
    
    def forward(self, data):
        x, edge_index, batch, area, perm = data.x, data.edge_index, data.batch, data.area, data.perm
        
        # print("input", x.shape, perm.shape)
        # print("batch size", batch.max() + 1) 
        
        global_features = torch.stack([perm], dim=1) # shape = batch, 1
        # print(type(x), type(edge_index), type(batch), type(area), type(perm))
        # print(x.shape, edge_index.shape, batch.shape, area.shape, perm.shape)
        # print(edge_index.max(), edge_index.min())
        # exit()
        x0 = self.initLin(x) # 32
        x1 = F.relu(self.conv1(x0, edge_index)) # 32
        x = torch.cat([x0, x1], dim=1) # 32 + 32 = 64
        x2 = F.relu(self.conv2(x, edge_index)) # 64
        x = torch.cat([x0, x1, x2], dim=1) # 32 + 32 + 64 = 128
        x3 = F.relu(self.conv3(x, edge_index)) # 128
        x = torch.cat([x0, x1, x2, x3], dim=1) # 32 + 32 + 64 + 128 = 256
        x = F.relu(self.conv4(x, edge_index)) # 64
        global_features = self.global_fc(global_features) # shape = batch, 16
        # print("global_features", global_features.shape)
        batchCounts = torch.bincount(batch)
        # print("batchCounts", batchCounts.shape)
        global_features = torch.repeat_interleave(global_features, batchCounts, dim=0).reshape(-1, 8)
        # print("global_features", global_features.shape)
        
        x = torch.cat([x, global_features], dim=1)
        # print("x", x.shape)
        
        batchedX, paddingMask = self.paddingToEachBatch(x, batch)
        # print("batchedX", batchedX.shape)
        # print("paddingMask", paddingMask.shape)
        x = self.transformer_encoder(batchedX, src_key_padding_mask=paddingMask)
        # print("after x", x.shape)
        x = self.att_pool(x, key_padding_mask=paddingMask)
        # print("end x", x.shape)
        
        return x
