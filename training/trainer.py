import os
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
            print(f"Epoch {epoch} | TrainLoss:{train_loss:.6f} | ValLoss:{val_metrics['val_loss']:.6f} | IC:{val_metrics['ic']:.4f} | RankIC:{val_metrics['rank_ic']:.4f}")
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
            print("保存最优模型")
