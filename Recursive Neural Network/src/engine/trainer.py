"""
Define general train/eval
"""
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from src.engine.metrics import MetricManager

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion, device, task_type='classification', task_name="dl_lab_03"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.task_type = task_type
        self.logger = logging.getLogger(task_name)


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            inputs = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(inputs)
            if self.task_type == 'ner':
                loss = self.criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            else:
                loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss/len(self.train_loader)
            self.logger.info(f"Finished one epoch with loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, loader=None, idx2label=None):
        self.model.eval()
        eval_loader = loader if loader is not None else self.val_loader

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in eval_loader:
                inputs = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(inputs)
                if self.task_type == "ner":
                    preds = torch.argmax(outputs, dim=-1) # (Batch, SeqLen) for each word
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(labels.cpu().tolist())
                else:
                    preds = torch.argmax(outputs, dim=1) #Batch for a sentence
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(labels.cpu().tolist())
        f1 = MetricManager.calculated_f1(all_preds, all_targets, self.task_type, idx2label)
        return f1