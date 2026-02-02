import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention).
    Calculates weights between current decoder hidden state and all encoder outputs.
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.W_enc = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W_dec = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: [batch_size, hid_dim] (trạng thái ẩn cuối của decoder)
        # enc_outputs: [batch_size, src_len, hid_dim]

        src_len = enc_outputs.size(1)

        # Lặp lại dec_hidden để khớp với src_len
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch, src_len, hid_dim]

        # Energy calculation: tanh(W_enc * enc_out + W_dec * dec_hid)
        energy = torch.tanh(self.W_enc(enc_outputs) + self.W_dec(dec_hidden))

        # Scores calculation
        scores = self.v(energy).squeeze(-1) # [batch, src_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Tính trọng số attention 
        attn_weights = F.softmax(scores, dim=1) # [batch, src_len]

        # Context vector: Weighted sum of enc_outputs
        # [batch, 1, src_len] * [batch, src_len, hid_dim] -> [batch, 1, hid_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

        return context, attn_weights