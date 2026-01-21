import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

from src.data.vsfc import VSFCDataset
from src.data.vocab import Vocab
from src.data.phoner import PhonerDataset
from src.models.gru import GRU
from src.models.lstm import LSTM
from src.models.bi_lstm import BiLSTM
from src.engine.trainer import Trainer
from src.utils.logger import setup_logger

def run_task(task_name):
    log_file = f"outputs/logs/{task_name}.log"
    logger = setup_logger(output_file=log_file, name=task_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN_SIZE = 256
    NUM_LAYERS = 5
    BATCH_SIZE = 36
    LR = 0.001
    EPOCHS = 2

    if task_name in ['lstm', 'gru']:
        logger.info(f"Loading VSFC Dataset for {task_name}")
        train_dataset = VSFCDataset("data/UIT-VSFC/UIT-VSFC-train.json", Vocab(), "topic")
        test_dataset = VSFCDataset("data/UIT-VSFC/UIT-VSFC-test.json", Vocab(), "topic")
        dev_dataset = VSFCDataset("data/UIT-VSFC/UIT-VSFC-dev.json", Vocab(), "topic")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        if task_name == 'lstm':
            logger.info("Initializing LSTM Classifier (5 layers)")
            model = LSTM(vocab=train_dataset.vocab)
        else:
            logger.info("Initializing GRU Classifier (5 layers)")
            model = GRU(vocab=train_dataset.vocab)
        task_type = 'classification'
        idx2label = train_dataset.vocab.idx2label

    elif task_name == 'bilstm':
        logger.info("Loading PhoNER Dataset")
        train_ds = PhonerDataset('data/PhoNER_COVID19/word/train_word.json')
        dev_ds = PhonerDataset('data/PhoNER_COVID19/word/dev_word.json', vocab=train_ds.vocab)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)

        logger.info("Initializing Bi-LSTM NER (5 layers)")
        model = BiLSTM()
        task_type = 'ner'
        idx2label = train_ds.vocab.idx2label
    else:
        logger.error(f"Task {task_name} is not found")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100 if task_type == 'ner' else -1)
    trainer = Trainer(
        model,
        train_loader,
        dev_loader,
        test_loader,
        optimizer=optimizer,
        criterion=criterion,
        task_type=task_type,
        task_name=task_name,
        device=device
    )
    logger.info(f"Strating training on {device}")
    for epoch in range(EPOCHS):
        loss = trainer.train_epoch()
        f1 = trainer.evaluate(loader=dev_loader,idx2label=idx2label)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Dev F1: {f1:.4f}")

    logger.info("FINAL EVALUATION ON TEST SET")
    test_f1 = trainer.evaluate(loader=test_loader, idx2label=idx2label)
    logger.info(f"FINAL TEST F1 SCORE: {test_f1:.4f}")
    logger.info("==================================================")

    checkpoint_dir = f"checkpoint/{task_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
    logger.info(f"Model saved to {checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lab 3 DL tasks")
    parser.add_argument('--task', type=str, required=True, help="Task name to run: lstm, gru, bilstm")
    args = parser.parse_args()
    run_task(args.task)