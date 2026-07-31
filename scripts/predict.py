import os
import sys
import yaml
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_interface import get_dataloaders
from inference.predictor import StockPredictor

def main(date_str):
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['data']
    model_cfg = config['model']
    
    _, test_loader, stock_sw_sector_map = get_dataloaders(
        seq_len=data_cfg['seq_len'],
        batch_size=data_cfg['batch_size']
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 只传模型构造函数需要的参数
    model_config = {
        'feature_dim': data_cfg['feature_dim'],
        'd_model': model_cfg['d_model'],
        'n_heads': model_cfg['n_heads'],
        'n_layers': model_cfg['n_layers'],
        'd_ff': model_cfg['d_ff'],
        'dropout': model_cfg['dropout'],
        'pred_len': data_cfg['pred_len'],
    }
    
    predictor = StockPredictor(
        checkpoint_path="outputs/checkpoints/best_model.pt",
        config=model_config,
        device=device
    )
    
    result = predictor.predict(test_loader, stock_sw_sector_map, None)
    
    os.makedirs("outputs/forecasts", exist_ok=True)
    output_path = f"outputs/forecasts/forecast_{date_str}.parquet"
    result.to_parquet(output_path)
    print(f"✅ 预测已保存: {output_path}")
    print(f"   样本数: {len(result)}")
    print(f"   预测收益率范围: [{result['predicted_return'].min():.6f}, {result['predicted_return'].max():.6f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260731")
    args = parser.parse_args()
    main(args.date)
