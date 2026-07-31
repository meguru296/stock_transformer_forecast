import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_transformer import build_transformer_dataloaders

def load_processed_data(data_dir="./data/raw/data_center/processed"):
    with open(os.path.join(data_dir, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    
    processed_features = {}
    for name in feature_names:
        processed_features[name] = pd.read_parquet(
            os.path.join(data_dir, f"feature_{name}.parquet")
        )
    
    macro_features = {
        'market_scaled': pd.read_parquet(os.path.join(data_dir, "macro_features.parquet"))
    }
    scaled_sector_ret = pd.read_parquet(os.path.join(data_dir, "sector_features.parquet"))
    
    target_y = pd.read_parquet(os.path.join(data_dir, "target_y.parquet"))
    target_mask = pd.read_parquet(os.path.join(data_dir, "target_mask.parquet"))
    label_dict = {'target_y': target_y, 'target_mask': target_mask}
    
    with open(os.path.join(data_dir, "stock_list.json"), "r", encoding="utf-8") as f:
        stocks = json.load(f)
    with open(os.path.join(data_dir, "stock_sw_sector_map.json"), "r", encoding="utf-8") as f:
        stock_sw_sector_map = json.load(f)
    
    print(f"数据加载: {len(processed_features)}个特征, {len(stocks)}只股票, {len(target_y)}个交易日")
    return processed_features, macro_features, scaled_sector_ret, label_dict, stock_sw_sector_map

def get_dataloaders(seq_len=30, batch_size=256, data_dir="./data/raw/data_center/processed"):
    processed_features, macro_features, scaled_sector_ret, label_dict, stock_sw_sector_map = load_processed_data(data_dir)
    
    train_loader, test_loader = build_transformer_dataloaders(
        aligned_features=None,
        processed_features=processed_features,
        mask_features=None,
        label_dict=label_dict,
        macro_features=macro_features,
        scaled_sector_ret=scaled_sector_ret,
        stock_sw_sector_map=stock_sw_sector_map,
        seq_len=seq_len,
        batch_size=batch_size
    )
    return train_loader, test_loader, stock_sw_sector_map

if __name__ == "__main__":
    train_loader, test_loader, _ = get_dataloaders()
    for batch_idx, (x, y, mask) in enumerate(train_loader):
        print(f"验证: X shape {x.shape}, Y shape {y.shape}, Mask shape {mask.shape}")
        break
