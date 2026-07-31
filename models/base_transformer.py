import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class VanillaTransformer(nn.Module):
    """
    基线Transformer时序预测模型
    输入: (batch, seq_len, feature_dim)
    输出: (batch,) 预测T+1对数收益率
    """
    def __init__(
        self,
        feature_dim=11,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        pred_len=1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        last_hidden = x[:, -1, :]
        out = self.output_projection(last_hidden)
        return out.squeeze(-1)

if __name__ == "__main__":
    model = VanillaTransformer(feature_dim=11, d_model=128, n_heads=8, n_layers=4)
    x = torch.randn(4, 30, 11)
    y = model(x)
    print(f"测试通过: 输入{x.shape} -> 输出{y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
