import torch
import torch.nn as nn

class DLinear(nn.Module):
    """
    DLinear: 简单线性分解模型
    将输入分解为趋势项和季节项，分别用线性层预测
    输入: (batch, seq_len, feature_dim)
    输出: (batch,) 预测T+1对数收益率
    """
    def __init__(
        self,
        feature_dim=11,
        seq_len=30,
        pred_len=1,
        individual=False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim
        
        # 趋势项线性层
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        # 季节项线性层
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        
        # 如果individual=True，每个变量独立一个线性层（这里简化用共享）
        
    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        
        # 简单移动平均提取趋势 (batch, seq_len, feature_dim)
        trend = x.mean(dim=1, keepdim=True).expand(-1, self.seq_len, -1)
        seasonal = x - trend
        
        # 转置为 (batch, feature_dim, seq_len) 以便线性层处理
        trend = trend.permute(0, 2, 1)      # (batch, feature_dim, seq_len)
        seasonal = seasonal.permute(0, 2, 1) # (batch, feature_dim, seq_len)
        
        # 线性预测
        trend_out = self.Linear_Trend(trend)      # (batch, feature_dim, pred_len)
        seasonal_out = self.Linear_Seasonal(seasonal)  # (batch, feature_dim, pred_len)
        
        # 合并
        out = trend_out + seasonal_out  # (batch, feature_dim, pred_len)
        
        # 对所有变量的预测取平均得到最终输出
        out = out.mean(dim=1)  # (batch, pred_len)
        
        return out.squeeze(-1)

if __name__ == "__main__":
    model = DLinear(feature_dim=11, seq_len=30)
    x = torch.randn(4, 30, 11)
    y = model(x)
    print(f"DLinear测试: 输入{x.shape} -> 输出{y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
