import os
import sys
import yaml
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_interface import get_dataset
from inference.predictor import StockPredictor

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    else: return "cpu"

def main():
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    model_cfg = config["model"]
    
    print("加载数据集...")
    dataset = get_dataset(seq_len=data_cfg["seq_len"])
    
    model_config = {
        "feature_dim": data_cfg["feature_dim"],
        "d_model": model_cfg["d_model"],
        "n_heads": model_cfg["n_heads"],
        "n_layers": model_cfg["n_layers"],
        "d_ff": model_cfg["d_ff"],
        "dropout": model_cfg["dropout"],
        "pred_len": data_cfg["pred_len"],
    }
    
    device = get_device()
    print(f"设备: {device}")
    
    predictor = StockPredictor(
        checkpoint_path="outputs/checkpoints/best_model.pt",
        config=model_config,
        device=device
    )
    
    print("\n1. 生成历史回测预测（测试集）...")
    history_df = predictor.predict_test_set(dataset, train_ratio=0.8, batch_size=data_cfg["batch_size"])
    print(f"   历史样本数: {len(history_df)}")
    
    print("\n2. 生成最新未来预测...")
    latest_df = predictor.predict_latest(dataset, batch_size=data_cfg["batch_size"])
    trade_date = latest_df["trade_date"].iloc[0]
    print(f"   最新预测日: {trade_date}")
    
    print("\n3. 合并...")
    combined = pd.concat([history_df, latest_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
    combined = combined.sort_values(["trade_date", "predicted_return"], ascending=[True, False])
    combined = combined.reset_index(drop=True)
    
    os.makedirs("outputs/forecasts", exist_ok=True)
    output_path = f"outputs/forecasts/forecast_full_{trade_date}.parquet"
    combined.to_parquet(output_path, index=False)
    
    print(f"\n✅ 完整预测已保存: {output_path}")
    print(f"   总行数: {len(combined)}")
    print(f"   交易日数: {combined['trade_date'].nunique()}")

if __name__ == "__main__":
    main()
