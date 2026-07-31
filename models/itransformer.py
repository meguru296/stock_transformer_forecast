import torch
import torch.nn as nn
import math

class iTransformer(nn.Module):
    """
    iTransformer: 在变量维度（而非时间步）做注意力
    把11个特征变量作为token，时间步作为序列长度
    输入: (batch, seq_len, feature_dim)
    输出: (batch,) 预测T+1对数收益率
    """
    def __init__(
        self,
        feature_dim=11,
        seq_len=30,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        pred_len=1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 每个变量独立嵌入到d_model
        self.embedding = nn.Linear(seq_len, d_model)
        
        # 位置编码（在变量维度上）
        pe = torch.zeros(1, feature_dim, d_model)
        position = torch.arange(0, feature_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer Encoder（在变量维度上做注意力）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出投影
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim * d_model, d_model),
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
        # 转置: (batch, feature_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 嵌入: (batch, feature_dim, d_model)
        x = self.embedding(x)
        x = x + self.pe
        
        # 在变量维度做自注意力
        x = self.transformer_encoder(x)  # (batch, feature_dim, d_model)
        
        # 展平并预测
        x = x.reshape(x.size(0), -1)
        out = self.head(x)
        return out.squeeze(-1)

if __name__ == "__main__":
    model = iTransformer(feature_dim=11, seq_len=30)
    x = torch.randn(4, 30, 11)
    y = model(x)
    print(f"iTransformer测试: 输入{x.shape} -> 输出{y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
