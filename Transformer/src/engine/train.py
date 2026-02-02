import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, task_type="classification"):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        # Masking cho Padding (để attention không nhìn vào <PAD>)
        mask = (inputs != 0).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, seq_len)

        optimizer.zero_grad()
        outputs = model(inputs, mask)

        if task_type == "ner":
            # Flatten outputs và labels cho CrossEntropyLoss
            # outputs: (batch, seq, num_labels) -> (batch * seq, num_labels)
            # labels: (batch, seq) -> (batch * seq)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)