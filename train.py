import torch
from torchview import draw_graph

from data.dataset import build_dataloaders
from data.generators import generate_data
from models.lstm_with_attention import LSTMWithAttention
from training.trainer import train_model

# Параметры модели
input_dim = 1
hidden_dim = 128
num_layers = 2
seq_len = 20
batch_size = 2
bidirectional = False

x = torch.randn(batch_size, seq_len)
mask = torch.ones(batch_size, seq_len).bool()

# Инициализация модели
model = LSTMWithAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1,
                          bidirectional=bidirectional)

print(f"Total parameters:     {model.count_parameters():,}")
print(f"Trainable parameters: {model.count_parameters(trainable_only=True):,}")

draw_graph(model, input_data=(x, mask), graph_name='LSTM with Attention', expand_nested=True,
           roll=True).visual_graph.render(format='svg')

data, labels, masks = generate_data(num_samples=250_000)
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Masks shape:", masks.shape)

train_loader, val_loader, test_loader = build_dataloaders(data, labels, masks, split=(0.8, 0.1, 0.1))

print('Sample of train_loader:')
print(next(iter(train_loader)))

print('Torch CUDA status: \n\t%s' %
      '✅ Available' if torch.cuda.is_available() else '❌ NOT available', '\n')

# Обучение
trained_model, train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=2_000,
    patience=20,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_every=25
)

torch.save({
    "config": trained_model.get_config(),
    "state_dict": trained_model.state_dict(),
}, "weights/lstm_epoch82_val005.pt")
torch.save(trained_model, "saved_models/lstm_full_v3.pt")

print("Model trained and saved successfully ✅")
