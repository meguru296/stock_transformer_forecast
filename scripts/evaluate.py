import os
import sys
import yaml
import torch
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_interface import get_dataloaders
from models.base_transformer import VanillaTransformer
from training.metrics import compute_metrics

def main():
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['data']
    model_cfg = config['model']
    
    _, test_loader, _ = get_dataloaders(
        seq_len=data_cfg['seq_len'],
        batch_size=data_cfg['batch_size']
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VanillaTransformer(
        feature_dim=data_cfg['feature_dim'],
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout'],
        pred_len=data_cfg['pred_len']
    ).to(device)
    
    checkpoint = torch.load("outputs/checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y)
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    metrics = compute_metrics(all_preds, all_targets)
    
    print("\n" + "="*50)
    print("📊 测试集评估结果")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v: .6f}")
    print("="*50)
    
    os.makedirs("outputs/reports", exist_ok=True)
    report = {
        "model": model_cfg['name'],
        "checkpoint_epoch": checkpoint.get('epoch', 'unknown'),
        "metrics": metrics
    }
    with open("outputs/reports/evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print("\n✅ 报告已保存: outputs/reports/evaluation_report.json")

if __name__ == "__main__":
    main()
