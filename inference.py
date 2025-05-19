import argparse
import torch
import numpy as np
from models.lstm_with_attention import LSTMWithAttention

def load_model(path="weights/lstm_epoch69_val040.pt", input_dim=1, hidden_dim=64):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))

    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    model = LSTMWithAttention(**config)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def predict_next(model, sequence, max_length=20):
    seq_len = len(sequence)
    if seq_len > max_length:
        raise ValueError(f"Sequence is longer than supported max length ({max_length})")

    x = torch.tensor(sequence, dtype=torch.float32)
    # mean = x.mean()
    # std = x.std() + 1e-8
    # x_norm = (x - mean) / std
    x_norm = x

    padding = max_length - seq_len
    padded_x = torch.cat([x_norm, torch.zeros(padding)], dim=0).unsqueeze(0)
    mask = torch.tensor([1] * seq_len + [0] * padding, dtype=torch.bool).unsqueeze(0)

    with torch.no_grad():
        output, attn = model(padded_x, mask)

    # predict normalization
    # pred = output.item() * std.item() + mean.item()
    pred = output.item()
    return pred, attn.squeeze().numpy()[:seq_len]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the next value in a numerical sequence.")
    parser.add_argument("sequence", nargs='+', help="Input sequence of numbers (space-separated)", type=float)
    parser.add_argument("--weights", type=str, default="weights/lstm_epoch69_val040.pt", help="Path to model weights")
    parser.add_argument("--max-length", type=int, default=20, help="Maximum sequence length")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = load_model(path=args.weights)
    pred, attn = predict_next(model, args.sequence, max_length=args.max_length)

    print("\nInput Sequence:", args.sequence)
    print("Predicted Next Value: {:.4f}".format(pred))
    print("Attention Weights:", [f"{w:.3f}" for w in attn])
