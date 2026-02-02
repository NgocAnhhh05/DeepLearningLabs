import torch
import torch.nn as nn
from src.data_processor.phomt import PhoMTDataset
from src.data_processor.vocab import Vocab
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.utils.logger import setup_logger

def main():
    logger = setup_logger("outputs/logs/lab04.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data & Build Vocabs
    logger.info("Loading Data...")
    train_raw_ds = PhoMTDataset("data/small-PhoMT/small-train.json")

    src_vocab = Vocab()
    src_vocab.build_vocab(train_raw_ds.src_sents)

    tgt_vocab = Vocab()
    tgt_vocab.build_vocab(train_raw_ds.tgt_sents)

    train_ds = PhoMTDataset("data/small-PhoMT/small-train.json", src_vocab, tgt_vocab)
    dev_ds = PhoMTDataset("data/small-PhoMT/small-dev.json", src_vocab, tgt_vocab)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=32)

    # 2. Models (3 Layers LSTM, 256 Hidden size)
    enc = Encoder(len(src_vocab.word2idx), 128, 256, 3, 0.5)
    dec = Decoder(len(tgt_vocab.word2idx), 128, 256, 3, 0.5)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.word2idx["<PAD>"])

    # 3. Training
    trainer = Trainer(model, train_loader, dev_loader, optimizer, criterion, device)
    evaluator = Evaluator(model, device, tgt_vocab)

    for epoch in range(10):
        loss = trainer.train_epoch()
        logger.info(f"Epoch {epoch+1} | Loss: {loss:.4f}")

        # Simple test translation
        sample_src = next(iter(dev_loader))['src_ids'][:1].to(device)
        translated = evaluator.translate(sample_src)
        logger.info(f"Sample Translation: {translated[0]}")

if __name__ == "__main__":
    main()