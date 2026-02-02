import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_processor.uit_viocd import ViOCDDataset
from src.data_processor.phonert import PhonerDataset
from src.data_processor.vocab import Vocab
from src.models.transformer import TransformerForClassification, TransformerForSequenceLabeling
from src.engine.train import train_one_epoch
from src.engine.evaluate import evaluate
from src.utils.logger import setup_logger

def run_task_1():
    logger = setup_logger(name="Task1_ViOCD")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab()
    train_ds = ViOCDDataset("data/uit_viocd/train.json", vocab, "topic")
    dev_ds = ViOCDDataset("data/uit_viocd/dev.json", vocab, "topic")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=train_ds.collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=dev_ds.collate_fn)

    model = TransformerForClassification(vocab.vocab_size, vocab.num_labels, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting Task 1: UIT-ViOCD Classification")
    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, dev_loader, device)
        logger.info(f"Epoch {epoch+1} - Loss: {loss:.4f} - Acc: {acc:.4f}")

def run_task_2():
    logger = setup_logger(name="Task2_Phonert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab()
    train_ds = ViOCDDataset("data/phonert/train.json", vocab, "topic")
    dev_ds = ViOCDDataset("data/phonert/dev.json", vocab, "topic")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=train_ds.collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=dev_ds.collate_fn)

    model = TransformerForClassification(vocab.vocab_size, vocab.num_labels, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting Task 2: Phonert Classification")
    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, dev_loader, device)
        logger.info(f"Epoch {epoch+1} - Loss: {loss:.4f} - Acc: {acc:.4f}")


if __name__ == "__main__":
    run_task_1()
    # run_task_2()