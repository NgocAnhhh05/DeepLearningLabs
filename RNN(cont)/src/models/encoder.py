import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder with 3-layer LSTM.
    Returns:
        outputs: All hidden states for each time step (used for Attention).
        hidden: Final hidden state of each layer.
        cell: Final cell state of each layer.
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src) # [batch_size, src_len, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim] -> All time steps
        # hidden/cell: [n_layers, batch_size, hid_dim]

        return outputs, hidden, cell

if __name__ == "__main__":
    # Test Encoder
    model = Encoder(100, 128, 256, 3, 0.5)
    test_input = torch.randint(0, 100, (8, 20))
    outputs, h, c = model(test_input)
    print(f"Outputs shape: {outputs.shape}") # [8, 20, 256]