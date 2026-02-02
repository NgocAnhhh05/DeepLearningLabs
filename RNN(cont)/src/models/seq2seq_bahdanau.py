import torch
import torch.nn as nn
import random
from .attention import BahdanauAttention

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = BahdanauAttention(encoder.hid_dim)
        self.device = device
        self.pad_idx = pad_idx

    def create_mask(self, src):
        mask = (src != self.pad_idx) # [batch, src_len]
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encoder forward
        enc_outputs, hidden, cell = self.encoder(src)
        mask = self.create_mask(src)

        # First input to decoder is <SOS>
        input = trg[:, 0]

        for t in range(1, trg_len):
            # 1. Calculate Attention context
            context, _ = self.attention(hidden[-1], enc_outputs, mask)

            # 2. Decoder step
            output, hidden, cell = self.decoder(input, hidden, cell, context)

            outputs[:, t, :] = output

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs