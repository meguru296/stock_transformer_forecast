import os
import sys
import yaml
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_interface import get_dataloaders
from models.base_transformer import VanillaTransformer
from training.trainer import Trainer

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']

    print("加载数据...")
    train_loader, test_loader, _ = get_dataloaders(
        seq_len=data_cfg['seq_len'],
        batch_size=data_cfg['batch_size']
    )

    print("构建模型...")
    model = VanillaTransformer(
        feature_dim=data_cfg['feature_dim'],
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        d_ff=model_cfg['d_ff'],
        dropout=model_cfg['dropout'],
        pred_len=data_cfg['pred_len']
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    device = get_device()
    print(f"设备: {device}")

    trainer = Trainer(model, train_cfg, device=device)
    history = trainer.fit(train_loader, test_loader)
    print("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
