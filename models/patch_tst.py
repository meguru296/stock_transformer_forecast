import torch
import torch.nn as nn
import math

class PatchTST(nn.Module):
    """
    PatchTST: 将时序分Patch后在Patch维度做自注意力
    输入: (batch, seq_len, feature_dim)
    输出: (batch,) 预测T+1对数收益率
    """
    def __init__(
        self,
        feature_dim=11,
        seq_len=30,
        patch_len=6,
        stride=3,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        pred_len=1
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # 计算patch数量
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Patch embedding: 每个变量独立处理
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # 位置编码 (patches维度)
        pe = torch.zeros(1, self.num_patches, d_model)
        position = torch.arange(0, self.num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer Encoder (在patch维度上做注意力)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出头 —— 修复：输入维度要乘 feature_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim * self.num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        
        # 对每个变量独立做patch
        x = x.permute(0, 2, 1)  # (batch, feature_dim, seq_len)
        
        # 分patch: (batch, feature_dim, num_patches, patch_len)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # 合并batch和feature维度: (batch*feature_dim, num_patches, patch_len)
        x = x.reshape(-1, self.num_patches, self.patch_len)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch*feature_dim, num_patches, d_model)
        x = x + self.pe
        
        # Transformer
        x = self.transformer_encoder(x)  # (batch*feature_dim, num_patches, d_model)
        
        # 恢复维度: (batch, feature_dim, num_patches, d_model)
        x = x.reshape(batch_size, self.feature_dim, self.num_patches, -1)
        
        # 展平: (batch, feature_dim * num_patches * d_model)
        x = x.reshape(batch_size, -1)
        
        # 输出
        out = self.head(x)
        return out.squeeze(-1)

if __name__ == "__main__":
    model = PatchTST(feature_dim=11, seq_len=30, patch_len=6, stride=3)
    x = torch.randn(4, 30, 11)
    y = model(x)
    print(f"PatchTST测试: 输入{x.shape} -> 输出{y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
