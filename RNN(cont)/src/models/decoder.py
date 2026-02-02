import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder with 3-layer LSTM and Attention integration.
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # RNN input: Embedding of current word
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        # Output layer: takes [dec_hidden; context]
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)

    def forward(self, input, hidden, cell, context):
        # input: [batch_size] (current word index)
        # context: [batch_size, hid_dim] (from attention)

        input = input.unsqueeze(1) # [batch, 1]
        embedded = self.embedding(input) # [batch, 1, emb_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # dec_hidden_last: lấy hidden state của layer cuối cùng
        dec_hidden_last = hidden[-1]

        # Concatenate hidden state and context vector
        combined = torch.cat((dec_hidden_last, context), dim=1) # [batch, hid_dim * 2]

        prediction = self.fc_out(combined) # [batch, output_dim]

        return prediction, hidden, cell