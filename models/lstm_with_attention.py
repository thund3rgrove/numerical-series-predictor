import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, mask=None):
        # encoder outputs: (batch_size, seq_len, hidden_dim)
        attn_scores = self.attn(encoder_outputs).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)
        return context, attn_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = Attention(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch_size, seq_len)
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_linear(x)  # (batch_size, seq_len, hidden_dim)

        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)

        context, attn_weights = self.attention(lstm_out, mask)
        output = self.output_linear(context)

        return output, attn_weights
