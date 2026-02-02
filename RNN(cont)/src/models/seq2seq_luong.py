import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LuongAttention(nn.Module):
    """
    Luong Attention (Global Attention)
    Paper: https://arxiv.org/abs/1508.04025
    """
    def __init__(self, hid_dim, attention_type="general"):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == "general":
            self.W = nn.Linear(hid_dim, hid_dim, bias=False)
        elif attention_type == "dot":
            self.W = None
        else:
            raise ValueError("attention_type must be 'dot' or 'general'")

    def forward(self, dec_output, enc_outputs, mask=None):
        """
        dec_output (ht): (batch, hid_dim) - Current hidden state of decoder
        enc_outputs (hs): (batch, src_len, hid_dim) - All hidden states of encoder
        """
        if self.attention_type == "general":
            dec_output = self.W(dec_output) # (batch, hid_dim)

        # score = ht * W * hs (General) or ht * hs (Dot)
        # Reshape dec_output for batch matrix multiplication
        # (batch, hid_dim, 1)
        dec_output = dec_output.unsqueeze(2)

        # scores: (batch, src_len, 1) -> (batch, src_len)
        scores = torch.bmm(enc_outputs, dec_output).squeeze(2)

        if mask is not None:
            # fill pad positions with very small value before softmax
            scores = scores.masked_fill(mask == 0, -1e10)

        # a_t: (batch, src_len)
        attn_weights = F.softmax(scores, dim=1)

        # context (ct) = sum(a_t * hs)
        # (batch, 1, src_len) * (batch, src_len, hid_dim) -> (batch, 1, hid_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

        return context, attn_weights

class Seq2SeqLuong(nn.Module):
    def __init__(self, encoder, decoder, device, tgt_vocab, attention_type="general"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.tgt_vocab = tgt_vocab # Đối tượng Vocab để lấy size và các ID đặc biệt

        self.attention = LuongAttention(encoder.hid_dim, attention_type)

        # h~t = tanh(Wc [ct ; ht])
        self.concat = nn.Linear(encoder.hid_dim * 2, encoder.hid_dim)

        # Layer cuối để dự đoán từ
        self.fc_out = nn.Linear(encoder.hid_dim, len(tgt_vocab.word2idx))

    def create_mask(self, src):
        # src: (batch, src_len)
        # Giả sử pad_id là 0 như các bài trước
        mask = (src != 0)
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_len), trg: (batch, trg_len)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = len(self.tgt_vocab.word2idx)

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encoder forward: lấy tất cả outputs để làm hs
        enc_outputs, hidden, cell = self.encoder(src)
        mask = self.create_mask(src)

        # Input đầu tiên của decoder là <SOS>
        input = trg[:, 0]

        for t in range(1, trg_len):
            # 1. Decoder RNN step
            # Note: Luong Attention thường dùng output của RNN tại bước t
            # decoder_output chính là hidden state lớp cuối
            output_step, hidden, cell = self.decoder(input, hidden, cell)

            # ht: hidden state lớp cuối cùng tại bước t (batch, hid_dim)
            ht = hidden[-1]

            # 2. Tính Context Vector dựa trên ht vừa tạo
            context, _ = self.attention(ht, enc_outputs, mask)

            # 3. Attentional Hidden State: h~t = tanh(Wc [ht ; ct])
            combined = torch.cat([ht, context], dim=1)
            h_tilde = torch.tanh(self.concat(combined))

            # 4. Prediction
            prediction = self.fc_out(h_tilde)
            outputs[:, t, :] = prediction

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

    def predict(self, src, max_len=50):
        """Dùng cho Evaluation (Greedy Decoding)"""
        self.eval()
        batch_size = src.size(0)

        with torch.no_grad():
            enc_outputs, hidden, cell = self.encoder(src)
            mask = self.create_mask(src)

            # Khởi tạo với <SOS>
            sos_id = self.tgt_vocab.word2idx["<SOS>"]
            eos_id = self.tgt_vocab.word2idx["<EOS>"]
            input = torch.full((batch_size,), sos_id, dtype=torch.long).to(self.device)

            all_predictions = []

            for _ in range(max_len):
                output_step, hidden, cell = self.decoder(input, hidden, cell)
                ht = hidden[-1]

                context, _ = self.attention(ht, enc_outputs, mask)
                combined = torch.cat([ht, context], dim=1)
                h_tilde = torch.tanh(self.concat(combined))

                prediction = self.fc_out(h_tilde)
                top1 = prediction.argmax(1)

                all_predictions.append(top1.unsqueeze(1))
                input = top1

            return torch.cat(all_predictions, dim=1)