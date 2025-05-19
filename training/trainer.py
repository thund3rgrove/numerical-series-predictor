import os
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, train_loader, val_loader, *,
                epochs=100, patience=10, lr=1e-3,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                checkpoint_every=50):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0

    train_losses = []
    val_losses = []

    # Создание директории для чекпоинтов
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = f"checkpoints/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

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

        # Save best state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch

        # Save checkpoint every N epochs
        if epoch % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch:03d}_best{best_epoch:03d}_valloss{best_val_loss:.4f}.pt")
            torch.save({
                "state_dict": best_state,
                "config": model.get_config(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch
            }, path)
            tqdm.write(f"Checkpoint saved at epoch {epoch} ✅")

        # Early Stopping
        if epoch - best_epoch >= patience:
            tqdm.write(f"\nEarly stopping triggered. Best epoch: {best_epoch}, val_loss: {best_val_loss:.6f}")
            break

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses
