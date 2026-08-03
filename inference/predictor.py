import os
import torch
import numpy as np
import pandas as pd

class StockPredictor:
    def __init__(self, checkpoint_path, config, device="cpu"):
        from models.base_transformer import VanillaTransformer
        self.device = device
        self.model = VanillaTransformer(**config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model_version = f"vanilla_transformer_v{checkpoint.get('epoch', 'x')}"

    @torch.no_grad()
    def predict_test_set(self, dataset, train_ratio=0.8, batch_size=256):
        split_idx = int(train_ratio * len(dataset.dates))
        test_samples = [(i, d, s) for i, (d, s) in enumerate(dataset.samples) if d >= split_idx]
        print(f"预测样本数: {len(test_samples)} (测试集, day_idx >= {split_idx})")
        return self._predict_samples(dataset, test_samples, batch_size)

    @torch.no_grad()
    def predict_latest(self, dataset, batch_size=256):
        max_day = max(d for d, s in dataset.samples)
        latest_samples = [(i, d, s) for i, (d, s) in enumerate(dataset.samples) if d == max_day]
        print(f"预测最新交易日: {dataset.dates[max_day].strftime('%Y-%m-%d')}, 样本数: {len(latest_samples)}")
        return self._predict_samples(dataset, latest_samples, batch_size)

    def _predict_samples(self, dataset, sample_indices, batch_size=256):
        predictions, dates, stock_codes, actual_returns = [], [], [], []
        for i in range(0, len(sample_indices), batch_size):
            batch = sample_indices[i:i+batch_size]
            batch_x, batch_y = [], []
            for sample_idx, day_idx, stock_idx in batch:
                x, y, mask = dataset[sample_idx]
                batch_x.append(x)
                batch_y.append(y)
                dates.append(dataset.dates[day_idx].strftime("%Y-%m-%d"))
                stock_codes.append(dataset.symbols[stock_idx])
            batch_x = torch.stack(batch_x).to(self.device)
            pred = self.model(batch_x).cpu().numpy()
            predictions.extend(pred.tolist())
            actual_returns.extend([float(y) for y in batch_y])
        result = pd.DataFrame({
            "trade_date": dates,
            "stock_code": stock_codes,
            "predicted_return": predictions,
            "actual_return": actual_returns,
            "confidence": np.abs(predictions),
            "direction_prob": (np.array(predictions) > 0).astype(float),
            "model_version": self.model_version,
            "inference_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        result = result.sort_values(["trade_date", "predicted_return"], ascending=[True, False])
        return result.reset_index(drop=True)
