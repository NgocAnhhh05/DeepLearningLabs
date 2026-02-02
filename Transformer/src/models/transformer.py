import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Tạo ma trận PE theo công thức trong paper
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        return torch.matmul(p_attn, v)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear projections và tách đầu (heads)
        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2. Apply attention
        x = self.attention(q, k, v, mask)

        # 3. Concatenate và linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Residual connection + LayerNorm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=3, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.pe(self.embedding(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Model cho Bài 1: Classification
class TransformerForClassification(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=512, n_layers=3):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)
        out = self.encoder(x, mask)
        # Pooling: lấy trung bình tất cả các token (hoặc có thể lấy token đầu tiên)
        out = torch.mean(out, dim=1)
        return self.classifier(out)

# Model cho Bài 2: Sequence Labeling (NER)
class TransformerForSequenceLabeling(nn.Module):
    def __init__(self, vocab_size, num_labels, d_model=512, n_layers=3):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x, mask=None):
        # out shape: (batch_size, seq_len, d_model)
        out = self.encoder(x, mask)
        # Dự đoán cho mỗi token
        return self.classifier(out)