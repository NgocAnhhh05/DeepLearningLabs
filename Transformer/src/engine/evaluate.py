import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader, device, task_type="classification"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            mask = (inputs != 0).unsqueeze(1).unsqueeze(2)

            outputs = model(inputs, mask)

            if task_type == "ner":
                preds = torch.argmax(outputs, dim=-1)
                # Chỉ lấy các nhãn không phải padding (-100)
                active_mask = (labels != -100)
                all_preds.extend(preds[active_mask].cpu().numpy())
                all_labels.extend(labels[active_mask].cpu().numpy())
            else:
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    if task_type == "ner":
        # Sử dụng F1 macro/weighted cho NER
        return f1_score(all_labels, all_preds, average='weighted')
    else:
        return accuracy_score(all_labels, all_preds)