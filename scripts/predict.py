import os
import sys
import yaml
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_interface import get_dataset
from inference.predictor import StockPredictor

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main(date_str):
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
        checkpoint_path="outputs/checkpoints/best_model_vanilla_transformer.pt",
        config=model_config, device=device
    )
    print("开始推理...")
    result = predictor.predict_test_set(dataset, train_ratio=0.8, batch_size=data_cfg["batch_size"])
    os.makedirs("outputs/forecasts", exist_ok=True)
    output_path = f"outputs/forecasts/forecast_{date_str}.parquet"
    result.to_parquet(output_path, index=False)
    print(f"\n预测已保存: {output_path}")
    print(f"   样本数: {len(result)}")
    print(f"   日期范围: {result['trade_date'].min()} ~ {result['trade_date'].max()}")
    print(f"   股票数: {result['stock_code'].nunique()}")
    print(f"   预测收益率范围: [{result['predicted_return'].min():.6f}, {result['predicted_return'].max():.6f}]")
    print("\n每日Top5预测示例:")
    for date, group in result.groupby("trade_date"):
        top5 = group.head(5)[["stock_code", "predicted_return", "confidence"]]
        print(f"\n{date}:")
        for _, row in top5.iterrows():
            print(f"  {row['stock_code']}: pred={row['predicted_return']:+.6f}, conf={row['confidence']:.6f}")
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260731")
    args = parser.parse_args()
    main(args.date)
