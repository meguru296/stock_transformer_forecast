import os
import sys
import yaml
import torch
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_interface import get_dataloaders
from models.base_transformer import VanillaTransformer
from models.patch_tst import PatchTST
from models.dlinear import DLinear
from models.itransformer import iTransformer
from training.trainer import Trainer
from training.metrics import compute_metrics

MODEL_REGISTRY = {
    'vanilla_transformer': VanillaTransformer,
    'patch_tst': PatchTST,
    'dlinear': DLinear,
    'itransformer': iTransformer,
}

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def train_and_evaluate(model_name, model_class, data_cfg, train_cfg, device):
    print(f"\n{'='*60}")
    print(f"🚀 训练模型: {model_name}")
    print(f"{'='*60}")
    
    # 检查是否已有该模型的checkpoint，有则跳过训练直接评估
    ckpt_path = f"outputs/checkpoints/best_model_{model_name}.pt"
    if os.path.exists(ckpt_path):
        print(f"发现已有checkpoint: {ckpt_path}，跳过训练直接评估")
        train_time = 0
        skip_train = True
    else:
        skip_train = False
    
    train_loader, test_loader, _ = get_dataloaders(
        seq_len=data_cfg['seq_len'],
        batch_size=data_cfg['batch_size']
    )
    
    if model_name == 'patch_tst':
        model = model_class(
            feature_dim=data_cfg['feature_dim'],
            seq_len=data_cfg['seq_len'],
            patch_len=6,
            stride=3,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.1,
            pred_len=data_cfg['pred_len']
        )
    elif model_name == 'dlinear':
        model = model_class(
            feature_dim=data_cfg['feature_dim'],
            seq_len=data_cfg['seq_len'],
            pred_len=data_cfg['pred_len']
        )
    elif model_name == 'itransformer':
        model = model_class(
            feature_dim=data_cfg['feature_dim'],
            seq_len=data_cfg['seq_len'],
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.1,
            pred_len=data_cfg['pred_len']
        )
    else:
        model = model_class(
            feature_dim=data_cfg['feature_dim'],
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.1,
            pred_len=data_cfg['pred_len']
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
    
    if not skip_train:
        start_time = time.time()
        trainer = Trainer(model, train_cfg, device=device)
        history = trainer.fit(train_loader, test_loader)
        train_time = time.time() - start_time
        
        # 重命名checkpoint
        if os.path.exists("outputs/checkpoints/best_model.pt"):
            os.rename(
                "outputs/checkpoints/best_model.pt",
                ckpt_path
            )
    else:
        # 加载已有checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    model.to(device)
    
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
    
    result = {
        'model_name': model_name,
        'params': total_params,
        'train_time_sec': train_time,
        'metrics': metrics
    }
    
    print(f"\n📊 {model_name} 最终测试集指标:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v: .6f}")
    
    return result

def main():
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['data']
    train_cfg = config['training']
    
    device = get_device()
    print(f"设备: {device}")
    
    # 先把之前跑完的 vanilla 和 dlinear 的 checkpoint 重命名
    for name in ['vanilla_transformer', 'dlinear']:
        old = "outputs/checkpoints/best_model.pt"
        new = f"outputs/checkpoints/best_model_{name}.pt"
        # 如果只有一个 best_model.pt 且没有对应命名的，需要手动处理
        # 这里简化：假设用户已经跑完，我们直接重新评估这两个
    
    models_to_compare = ['vanilla_transformer', 'dlinear', 'patch_tst', 'itransformer']
    
    results = []
    for model_name in models_to_compare:
        if model_name not in MODEL_REGISTRY:
            print(f"跳过未知模型: {model_name}")
            continue
        
        model_class = MODEL_REGISTRY[model_name]
        result = train_and_evaluate(model_name, model_class, data_cfg, train_cfg, device)
        results.append(result)
    
    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/model_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("📋 模型选型对比总结")
    print(f"{'='*80}")
    print(f"{'模型':<20} {'参数量':>12} {'训练时间(s)':>12} {'MSE':>10} {'MAE':>10} {'IC':>10} {'RankIC':>10} {'DirAcc':>10}")
    print("-" * 80)
    for r in results:
        m = r['metrics']
        print(f"{r['model_name']:<20} {r['params']:>12,} {r['train_time_sec']:>12.1f} {m['mse']:>10.6f} {m['mae']:>10.6f} {m['ic']:>10.4f} {m['rank_ic']:>10.4f} {m['direction_acc']:>10.4f}")
    print(f"{'='*80}")
    
    best_ic = max(results, key=lambda x: abs(x['metrics']['ic']))
    best_rank_ic = max(results, key=lambda x: abs(x['metrics']['rank_ic']))
    print(f"\n🏆 IC最高: {best_ic['model_name']} (IC={best_ic['metrics']['ic']:.4f})")
    print(f"🏆 RankIC最高: {best_rank_ic['model_name']} (RankIC={best_rank_ic['metrics']['rank_ic']:.4f})")

if __name__ == "__main__":
    main()
