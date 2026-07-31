import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

class StockPredictor:
    def __init__(self, checkpoint_path, config, device='cpu'):
        from models.base_transformer import VanillaTransformer
        self.device = device
        self.model = VanillaTransformer(**config).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.model_version = f"{config.get('name','model')}_ep{checkpoint.get('epoch','x')}"
    
    @torch.no_grad()
    def predict(self, data_loader, stock_list, dates_index):
        predictions = []
        confidences = []
        actual_dates = []
        actual_stocks = []
        
        for x, y, mask in data_loader:
            x = x.to(self.device)
            pred = self.model(x)
            
            predictions.append(pred.cpu().numpy())
            confidences.append(torch.abs(pred).cpu().numpy())
        
        predictions = np.concatenate(predictions)
        confidences = np.concatenate(confidences)
        
        # 构建输出DataFrame（简化版，按sample顺序）
        result = pd.DataFrame({
            'predicted_return': predictions,
            'confidence': confidences,
            'direction_prob': (predictions > 0).astype(float),
            'model_version': self.model_version,
            'inference_time': pd.Timestamp.now()
        })
        
        return result
