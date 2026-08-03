import os
import sys
import yaml
import torch
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
    # 找到实际有样本的最后一天（即可预测T+1的最后一天）
    max_day = max(d for d, s in dataset.samples)
    trade_date = dataset.dates[max_day].strftime("%Y-%m-%d")
    target_date = dataset.dates[max_day + 1].strftime("%Y-%m-%d") if max_day + 1 < len(dataset.dates) else "未来"
    print(f"预测日 (T): {trade_date}")
    print(f"目标日 (T+1): {target_date}")
    model_config = {"feature_dim": data_cfg["feature_dim"], "d_model": model_cfg["d_model"], "n_heads": model_cfg["n_heads"], "n_layers": model_cfg["n_layers"], "d_ff": model_cfg["d_ff"], "dropout": model_cfg["dropout"], "pred_len": data_cfg["pred_len"]}
    device = get_device()
    print(f"设备: {device}")
    predictor = StockPredictor(checkpoint_path="outputs/checkpoints/best_model.pt", config=model_config, device=device)
    print("开始推理T+1...")
    result = predictor.predict_latest(dataset, batch_size=data_cfg["batch_size"])
    os.makedirs("outputs/forecasts", exist_ok=True)
    output_path = f"outputs/forecasts/forecast_{trade_date}.parquet"
    result.to_parquet(output_path, index=False)
    print(f"\nT+1预测已保存: {output_path}")
    print(f"预测日: {trade_date} (预测 {target_date} 的收益率)")
    print(f"股票数: {len(result)}")
    print(f"\nTop5 推荐:")
    print(result.nlargest(5, "predicted_return")[["stock_code", "predicted_return", "confidence"]])

if __name__ == "__main__":
    main()
