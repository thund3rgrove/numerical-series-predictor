import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        # self.attn = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.Tanh(),
            # nn.Linear(hidden_dim // 2, 1)
        # )
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Identity()
        self.temperature = 1.0

    def forward(self, encoder_outputs, mask=None):
        attn_scores = self.attn(encoder_outputs).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores / self.temperature, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1, bidirectional=False):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        attn_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = Attention(attn_input_dim, dropout=dropout)
        self.output_linear = nn.Linear(attn_input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = x.unsqueeze(-1)
        x = self.input_linear(x)
        x = self.ln(x)
        x = F.relu(x)

        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out, mask)
        output = self.output_linear(context)

        return output, attn_weights

    def count_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_config(self):
        return {
            "input_dim": self.input_linear.in_features,
            "hidden_dim": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
            "dropout": self.lstm.dropout,
            "bidirectional": self.lstm.bidirectional
        }
