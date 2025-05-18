import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, train_loader, val_loader, *,
                epochs=100, patience=10, lr=1e-3,
                device='cuda' if torch.cuda.is_available() else 'cpu'):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_state = None
    early_stopping_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False)
        for x, mask, y in train_bar:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x, mask)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"[Epoch {epoch}] Validation", leave=False)
        with torch.no_grad():
            for x, mask, y in val_bar:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                output, _ = model(x, mask)
                loss = criterion(output.squeeze(), y)
                val_loss += loss.item() * x.size(0)
                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Обновление общей полоски
        tqdm.write(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch == 1:
            print(f"Example input: {x[0, :mask[0].sum()].cpu().numpy()}")
            print(f"Example target: {y[0].item()}")
            print(f"Example prediction: {output[0].item()}")


        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                tqdm.write(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses
