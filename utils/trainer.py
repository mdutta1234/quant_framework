# utils/trainer.py
import torch
import os

class MultiClusterTrainer:
    def __init__(self, models: dict, dataloaders: dict, loss_fn, config: dict):
        self.models = models
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        
        pred_config = config.get('predictor', {})
        self.epochs = pred_config.get('epochs', 20)
        self.lr = pred_config.get('learning_rate', 0.001)
        
        # Automatically use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup independent optimizers for each cluster's model
        self.optimizers = {
            c_id: torch.optim.Adam(model.parameters(), lr=self.lr)
            for c_id, model in self.models.items()
        }
        
        for model in self.models.values():
            model.to(self.device)

    def train(self):
        print(f"\n[TRAINER] Starting training on {self.device} for {self.epochs} epochs.")
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # 1. Training Phase
            train_loss = self._run_epoch('train')
            
            # 2. Validation Phase
            val_loss = self._run_epoch('val')
            
            print(f"Epoch {epoch+1:03d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # 3. Model Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoints()
                print(f"  -> Found new best models! Saved to disk. (Val Loss: {best_val_loss:.4f})")

    def _run_epoch(self, phase: str) -> float:
        is_train = (phase == 'train')
        
        for model in self.models.values():
            model.train() if is_train else model.eval()
            
        total_loss = 0.0
        total_samples = 0
        
        for batch_x, batch_id, batch_cid, batch_y, batch_w in self.dataloaders[phase]:
            batch_x = batch_x.to(self.device)
            batch_id = batch_id.to(self.device)
            batch_cid = batch_cid.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_w = batch_w.to(self.device)
            
            # Route data to the correct cluster-specific model
            for c_id, model in self.models.items():
                mask = (batch_cid == c_id)
                if not mask.any():
                    continue  # No stocks for this cluster in this batch
                    
                x_sub = batch_x[mask]
                id_sub = batch_id[mask]
                y_sub = batch_y[mask]
                w_sub = batch_w[mask]
                
                if is_train:
                    self.optimizers[c_id].zero_grad()
                    
                with torch.set_grad_enabled(is_train):
                    preds = model(x_sub, id_sub)
                    loss = self.loss_fn(preds, y_sub, w_sub)
                    
                    if is_train:
                        loss.backward()
                        self.optimizers[c_id].step()
                        
                total_loss += loss.item() * x_sub.size(0)
                total_samples += x_sub.size(0)
                
        return total_loss / max(total_samples, 1)
        
    def _save_checkpoints(self):
        save_dir = "./data/models"
        os.makedirs(save_dir, exist_ok=True)
        for c_id, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(save_dir, f"cluster_{c_id}_best.pth"))