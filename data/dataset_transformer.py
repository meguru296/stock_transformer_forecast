import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class TransformerStockDataset(Dataset):
    """
    针对 Transformer 模型的单标的时序切片 Dataset
    输入形状: (Seq_Len, Feature_Dim)
    涵盖：[个股标准化因子 + 全局大盘宏观因子 + 个股专属申万一级行业因子]
    """
    def __init__(
        self, 
        processed_features, 
        macro_features, 
        scaled_sector_ret, 
        label_dict, 
        stock_sw_sector_map=None, 
        seq_len=30
    ):
        self.seq_len = seq_len
        
        # 1. 获取基础维度信息
        sample_df = list(processed_features.values())[0]
        self.dates = sample_df.index
        self.symbols = list(sample_df.columns)
        self.num_days = len(self.dates)
        self.num_stocks = len(self.symbols)
        
        # 2. 提取宏观特征 (Days, Macro_Dim)
        market_scaled = macro_features.get('market_scaled', pd.DataFrame(index=self.dates))
        macro_arr = market_scaled.values if not market_scaled.empty else np.zeros((self.num_days, 0))
        macro_3d = np.repeat(macro_arr[:, np.newaxis, :], self.num_stocks, axis=1) # (Days, Stocks, K_macro)
        
        # 3. 提取个股专属申万一级行业特征 (Days, Stocks, K_sector)
        if isinstance(scaled_sector_ret, pd.DataFrame) and not scaled_sector_ret.empty:
            num_sector_feats = 1  # 申万一级行业收益率
            sector_3d = np.zeros((self.num_days, self.num_stocks, num_sector_feats), dtype=np.float32)
            sector_mean = scaled_sector_ret.mean(axis=1).values  # 全行业均值保底
            
            for s_idx, symbol in enumerate(self.symbols):
                sw_sec_name = stock_sw_sector_map.get(symbol) if stock_sw_sector_map else None
                if sw_sec_name and (sw_sec_name in scaled_sector_ret.columns):
                    sector_3d[:, s_idx, 0] = scaled_sector_ret[sw_sec_name].values
                else:
                    sector_3d[:, s_idx, 0] = sector_mean
        else:
            sector_3d = np.zeros((self.num_days, self.num_stocks, 0), dtype=np.float32)

        # 4. 构建 3D 股票特征张量 (Days, Stocks, K_stock)
        feature_keys = sorted(list(processed_features.keys()))
        stock_feats_list = [processed_features[k].values for k in feature_keys]
        shapes = [arr.shape for arr in stock_feats_list]
        assert all(s == shapes[0] for s in shapes), f"形状不一致: {shapes}"
        stock_3d = np.stack(stock_feats_list, axis=-1)
        
        # 5. 三维特征融合 (Days, Stocks, Total_Features)
        # Total_Features = K_stock + K_macro + K_sector
        self.full_tensor = np.concatenate([stock_3d, macro_3d, sector_3d], axis=-1).astype(np.float32)
        
        # 6. 标签与 Mask 矩阵
        self.target_y = label_dict['target_y'].values.astype(np.float32)       # (Days, Stocks)
        self.target_mask = label_dict['target_mask'].values.astype(np.float32) # (Days, Stocks)
        
        # 7. 构建有效的样本索引列表 [(day_idx, stock_idx), ...]
        self.samples = []
        for d in range(self.seq_len - 1, self.num_days - 1): # -1 留给 T+1 预测
            for s in range(self.num_stocks):
                if self.target_mask[d, s] > 0.5: # 仅保留有效开盘交易日样本
                    self.samples.append((d, s))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        day_idx, stock_idx = self.samples[idx]
        
        # 切取过去 seq_len 天的时序特征: (Seq_Len, Feature_Dim)
        x_seq = self.full_tensor[day_idx - self.seq_len + 1 : day_idx + 1, stock_idx, :]

        # 切取对应时间窗口的 mask（1=有效交易，0=停牌/未上市）
        mask_seq = self.target_mask[day_idx - self.seq_len + 1 : day_idx + 1, stock_idx]

        # mask_seq 形状 (seq_len,)，扩展为 (seq_len, 1) 以广播乘到特征维度
        x_seq = x_seq * mask_seq[:, np.newaxis]  # 逐元素相乘，停牌日特征变为 0

        # 获取 T+1 目标收益率
        y_label = self.target_y[day_idx, stock_idx]

        

        return (
        torch.tensor(x_seq, dtype=torch.float32),
        torch.tensor(y_label, dtype=torch.float32),
        torch.tensor(mask_seq, dtype=torch.float32)  # 返回形状: (seq_len,)
    )


def build_transformer_dataloaders(
    aligned_features, 
    processed_features, 
    mask_features, 
    label_dict, 
    macro_features, 
    scaled_sector_ret, 
    stock_sw_sector_map=None,
    seq_len=30, 
    batch_size=256
):
    """构建 Transformer 训练与测试 DataLoader"""
    dataset = TransformerStockDataset(
        processed_features=processed_features,
        macro_features=macro_features,
        scaled_sector_ret=scaled_sector_ret,
        label_dict=label_dict,
        stock_sw_sector_map=stock_sw_sector_map,  # 👈 传入申万行业映射
        seq_len=seq_len
    )
    
    # -----------------------------------------------------------------
    # ⚠️ 关键修复：采用【按时间顺序 (Time-Series Split)】切分 Train / Test
    # 严禁使用 random_split，防止金融时序未来信息泄漏！
    # -----------------------------------------------------------------
    # 获取所有日期
    dates = dataset.dates
    split_date_idx = int(0.8 * len(dates))

    # 根据样本中的 day_idx 是否小于 split_date_idx 来划分
    train_indices = []
    test_indices = []
    for i, (d, s) in enumerate(dataset.samples):
        if d < split_date_idx:
            train_indices.append(i)
        else:
            test_indices.append(i)

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print(f"✅ [Transformer Dataset] 构建完成! 样本总数: {len(dataset)}, Train Batches: {len(train_loader)}, Test Batches: {len(test_loader)}")
    return train_loader, test_loader