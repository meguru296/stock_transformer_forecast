import os

# ========== 1. config ==========
with open("config/model_config.yaml", "w") as f:
    f.write("""data:
  upstream_repo: "ZIN99606/stock_daily_crawler"
  data_dir: "./data/raw"
  seq_len: 30
  pred_len: 5
  batch_size: 256
  num_stocks: 22
  feature_dim: 11

model:
  name: "vanilla_transformer"
  d_model: 128
  n_heads: 8
  n_layers: 4
  d_ff: 512
  dropout: 0.1
  activation: "gelu"

training:
  epochs: 100
  lr: 0.0001
  weight_decay: 0.00001
  early_stop_patience: 15
  loss: "huber"
  huber_delta: 0.1
  gradient_clip: 1.0

evaluation:
  train_ratio: 0.8
  metrics: ["mse", "mae", "ic", "rank_ic", "direction_acc"]
""")
print("1. config/model_config.yaml 更新 (pred_len: 5)")

# ========== 2. patch dataset_transformer.py ==========
with open("data/dataset_transformer.py", "r") as f:
    content = f.read()

content = content.replace(
    """        self.target_y = label_dict['target_y']
        self.target_mask = label_dict['target_mask']""",
    """        if 'target_y_multi' in label_dict:
            self.target_y = label_dict['target_y_multi']
            self.target_mask = label_dict['target_mask_multi']
            self.multi_step = True
        else:
            self.target_y = label_dict['target_y']
            self.target_mask = label_dict['target_mask']
            self.multi_step = False"""
)

content = content.replace(
    "        y = self.target_y.values[day_idx, stock_idx]",
    """        if self.multi_step:
            y = self.target_y[day_idx, stock_idx, :]
        else:
            y = self.target_y.values[day_idx, stock_idx]"""
)

with open("data/dataset_transformer.py", "w") as f:
    f.write(content)
print("2. data/dataset_transformer.py 已打多步补丁")

# ========== 3. data_interface.py ==========
with open("data/data_interface.py", "w") as f:
    f.write("""import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_transformer import build_transformer_dataloaders, TransformerStockDataset

def load_processed_data(data_dir="./data/raw/data_center/processed"):
    with open(os.path.join(data_dir, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    processed_features = {}
    for name in feature_names:
        processed_features[name] = pd.read_parquet(os.path.join(data_dir, f"feature_{name}.parquet"))
    macro_features = {"market_scaled": pd.read_parquet(os.path.join(data_dir, "macro_features.parquet"))}
    scaled_sector_ret = pd.read_parquet(os.path.join(data_dir, "sector_features.parquet"))
    
    target_y = pd.read_parquet(os.path.join(data_dir, "target_y.parquet"))
    target_mask = pd.read_parquet(os.path.join(data_dir, "target_mask.parquet"))
    
    multi_target = np.stack([
        target_y.shift(-1).values,
        target_y.shift(-2).values,
        target_y.shift(-3).values,
        target_y.shift(-4).values,
        target_y.shift(-5).values,
    ], axis=-1)
    
    multi_mask = ~np.isnan(multi_target).any(axis=-1)
    multi_mask = multi_mask & (target_mask.values == 1)
    
    label_dict = {
        "target_y": target_y,
        "target_mask": target_mask,
        "target_y_multi": multi_target,
        "target_mask_multi": multi_mask,
    }
    
    with open(os.path.join(data_dir, "stock_list.json"), "r", encoding="utf-8") as f:
        stocks = json.load(f)
    with open(os.path.join(data_dir, "stock_sw_sector_map.json"), "r", encoding="utf-8") as f:
        stock_sw_sector_map = json.load(f)
    print(f"数据加载: {len(processed_features)}个特征, {len(stocks)}只股票, {len(target_y)}个交易日, 多步目标: 5天")
    return processed_features, macro_features, scaled_sector_ret, label_dict, stock_sw_sector_map

def get_dataloaders(seq_len=30, batch_size=256, data_dir="./data/raw/data_center/processed"):
    processed_features, macro_features, scaled_sector_ret, label_dict, stock_sw_sector_map = load_processed_data(data_dir)
    train_loader, test_loader = build_transformer_dataloaders(
        aligned_features=None, processed_features=processed_features, mask_features=None,
        label_dict=label_dict, macro_features=macro_features, scaled_sector_ret=scaled_sector_ret,
        stock_sw_sector_map=stock_sw_sector_map, seq_len=seq_len, batch_size=batch_size)
    return train_loader, test_loader, stock_sw_sector_map

def get_dataset(seq_len=30, data_dir="./data/raw/data_center/processed"):
    processed_features, macro_features, scaled_sector_ret, label_dict, stock_sw_sector_map = load_processed_data(data_dir)
    dataset = TransformerStockDataset(
        processed_features=processed_features, macro_features=macro_features,
        scaled_sector_ret=scaled_sector_ret, label_dict=label_dict,
        stock_sw_sector_map=stock_sw_sector_map, seq_len=seq_len)
    return dataset
""")
print("3. data/data_interface.py 更新 (多步标签)")

# ========== 4. base_transformer.py ==========
with open("models/base_transformer.py", "w") as f:
    f.write("""import torch
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
    def __init__(self, feature_dim=11, d_model=128, n_heads=8, n_layers=4, d_ff=512, dropout=0.1, pred_len=5):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, pred_len))
        self._init_weights()
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_projection(x[:, -1, :])

if __name__ == "__main__":
    model = VanillaTransformer(pred_len=5)
    x = torch.randn(4, 30, 11)
    y = model(x)
    print(f"测试: {x.shape} -> {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
""")
print("4. models/base_transformer.py 更新 (pred_len=5)")

# ========== 5. loss.py ==========
with open("training/loss.py", "w") as f:
    f.write("""import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=0.1, weights=None):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        if weights is None:
            weights = [0.4, 0.25, 0.15, 0.12, 0.08]
        self.weights = torch.tensor(weights)
    def forward(self, pred, target, mask=None):
        loss = self.huber(pred, target)
        w = self.weights.to(pred.device)
        return (loss * w).mean()
""")
print("5. training/loss.py 更新 (多步加权)")

# ========== 6. metrics.py ==========
with open("training/metrics.py", "w") as f:
    f.write("""import numpy as np
import torch
from scipy.stats import spearmanr

def compute_metrics(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)
    results = {}
    for i in range(pred.shape[1]):
        p, t = pred[:, i], target[:, i]
        valid = ~(np.isnan(p) | np.isnan(t))
        p, t = p[valid], t[valid]
        if len(p) < 2:
            continue
        results[f"d{i+1}_mse"] = float(np.mean((p-t)**2))
        results[f"d{i+1}_mae"] = float(np.mean(np.abs(p-t)))
        results[f"d{i+1}_ic"] = float(np.corrcoef(p, t)[0,1])
        results[f"d{i+1}_rank_ic"] = float(spearmanr(p, t)[0])
        results[f"d{i+1}_dir_acc"] = float(np.mean((p>0)==(t>0)))
    results["avg_ic"] = np.mean([v for k,v in results.items() if k.endswith("_ic")])
    return results
""")
print("6. training/metrics.py 更新 (多步指标)")

# ========== 7. trainer.py ==========
with open("training/trainer.py", "w") as f:
    f.write("""import os
import json
import torch
from training.loss import HuberLoss
from training.metrics import compute_metrics

class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.criterion = HuberLoss(delta=config.get('huber_delta', 0.1))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_dir = "./outputs/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y, mask in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss, all_preds, all_targets = 0, [], []
        for x, y, mask in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            total_loss += self.criterion(pred, y).item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        metrics['val_loss'] = avg_loss
        return metrics

    def fit(self, train_loader, val_loader):
        history = []
        for epoch in range(1, self.config['epochs'] + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_metrics['epoch'] = epoch
            val_metrics['train_loss'] = train_loss
            history.append(val_metrics)
            d1_ic = val_metrics.get('d1_ic', 0)
            print(f"Epoch {epoch} | TrainLoss:{train_loss:.6f} | ValLoss:{val_metrics['val_loss']:.6f} | D1_IC:{d1_ic:.4f}")
            self.scheduler.step()
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stop_patience']:
                    print(f"早停触发, best_val_loss={self.best_val_loss:.6f}")
                    break
        with open(f"{self.checkpoint_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        return history

    def save_checkpoint(self, epoch, metrics, is_best=False):
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'metrics': metrics, 'config': self.config}
        torch.save(state, f"{self.checkpoint_dir}/model_epoch_{epoch}.pt")
        if is_best:
            torch.save(state, f"{self.checkpoint_dir}/best_model.pt")
            print(f"保存最优模型")
""")
print("7. training/trainer.py 更新")

# ========== 8. predictor.py ==========
with open("inference/predictor.py", "w") as f:
    f.write("""import os
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
    def predict_latest(self, dataset, batch_size=256):
        max_day = max(d for d, s in dataset.samples)
        latest_samples = [(i, d, s) for i, (d, s) in enumerate(dataset.samples) if d == max_day]
        print(f"预测最新交易日: {dataset.dates[max_day].strftime('%Y-%m-%d')}, 样本数: {len(latest_samples)}")
        return self._predict_samples(dataset, latest_samples, batch_size)

    def _predict_samples(self, dataset, sample_indices, batch_size=256):
        predictions, dates, stock_codes = [], [], []
        for i in range(0, len(sample_indices), batch_size):
            batch = sample_indices[i:i+batch_size]
            batch_x = []
            for sample_idx, day_idx, stock_idx in batch:
                x, y, mask = dataset[sample_idx]
                batch_x.append(x)
                dates.append(dataset.dates[day_idx].strftime("%Y-%m-%d"))
                stock_codes.append(dataset.symbols[stock_idx])
            batch_x = torch.stack(batch_x).to(self.device)
            pred = self.model(batch_x).cpu().numpy()
            predictions.append(pred)
        predictions = np.concatenate(predictions, axis=0)
        data = {"trade_date": dates, "stock_code": stock_codes, "model_version": self.model_version, "inference_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        for i in range(predictions.shape[1]):
            data[f"pred_return_d{i+1}"] = predictions[:, i]
            data[f"confidence_d{i+1}"] = np.abs(predictions[:, i])
            data[f"direction_d{i+1}"] = (predictions[:, i] > 0).astype(int)
        result = pd.DataFrame(data)
        result = result.sort_values(["trade_date", "pred_return_d1"], ascending=[True, False])
        return result.reset_index(drop=True)
""")
print("8. inference/predictor.py 更新")

# ========== 9. predict_future.py ==========
with open("scripts/predict_future.py", "w") as f:
    f.write("""import os
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
    latest_date = dataset.dates[-1].strftime("%Y-%m-%d")
    print(f"数据最新日期: {latest_date}")
    print("预测目标: T+1 ~ T+5 (未来5个交易日)")
    model_config = {"feature_dim": data_cfg["feature_dim"], "d_model": model_cfg["d_model"], "n_heads": model_cfg["n_heads"], "n_layers": model_cfg["n_layers"], "d_ff": model_cfg["d_ff"], "dropout": model_cfg["dropout"], "pred_len": data_cfg["pred_len"]}
    device = get_device()
    print(f"设备: {device}")
    predictor = StockPredictor(checkpoint_path="outputs/checkpoints/best_model.pt", config=model_config, device=device)
    print("开始推理未来5天...")
    result = predictor.predict_latest(dataset, batch_size=data_cfg["batch_size"])
    os.makedirs("outputs/forecasts", exist_ok=True)
    output_path = f"outputs/forecasts/forecast_future5_{latest_date}.parquet"
    result.to_parquet(output_path, index=False)
    print(f"\\n未来5天预测已保存: {output_path}")
    print(f"股票数: {len(result)}")
    print(f"\\nT+1 Top5:")
    print(result.nlargest(5, "pred_return_d1")[["stock_code", "pred_return_d1", "confidence_d1"]])

if __name__ == "__main__":
    main()
""")
print("9. scripts/predict_future.py 更新")

print("\n" + "="*50)
print("全部文件更新完成！")
print("="*50)
print("下一步: python3 scripts/train.py  (重新训练T+5模型)")
print("训练完: python3 scripts/predict_future.py")
