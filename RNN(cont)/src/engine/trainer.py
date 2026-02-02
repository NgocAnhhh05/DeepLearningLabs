import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for batch in tqdm(self.train_loader):
            src = batch['src_ids'].to(self.device)
            trg = batch['tgt_ids'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(src, trg)

            # output: [batch, trg_len, vocab_size], trg: [batch, trg_len]
            # Skip SOS for loss
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)