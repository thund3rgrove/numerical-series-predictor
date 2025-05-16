Numerical Series Predictor
=======================
## Overview
This project implements a sequence prediction model based on **LSTM with attention**, designed to predict the next value in various types of numerical sequences. It generalizes across different patterns like linear, sinusoidal, logarithmic, and noisy sequences.

## ✨ Capabilities
- Predict the next value in a sequence with varying structure
- Interpret predictions via attention mechanism
- Handle variable-length input sequences (up to a max)

## 💡 Supported Sequence Types:
The training dataset includes examples of:
- ✅ Linear sequences (e.g., arithmetic progressions)
- ✅ Noisy sequences
- ✅ Sinusoidal patterns
- ✅ Logarithmic decay curves
- ✅ Mixed or oscillatory sequences

> Data is procedurally generated and can be extended.


## 📁 Project Structure
```
project_root/
├── models/             # LSTM + Attention architecture
├── inference/          # Inference script with CLI
├── weights/            # Trained state_dict model weights (.pt)
├── saved_models/       # Full saved model objects
├── generate_data.py    # Synthetic sequence generator
├── train_loop.py       # Training loop with early stopping
├── test_and_visualize.py # Evaluation utilities
├── main.ipynb          # Jupyter Notebook for experimentation
```

---

## 🧪 Training Pipeline

### 🔹 Data Generation
```python
from generate_data import generate_dataset

data, labels, masks = generate_dataset(num_samples=100_000)
```

### 🔹 Model Training
```python
from train_loop import train_model
from models.lstm_with_attention import LSTMWithAttention

model = LSTMWithAttention()
trained_model = train_model(model, train_loader, val_loader)
torch.save(trained_model.state_dict(), "weights/lstm_with_attention_best.pt")
```

### 🔹 Testing
```python
from test_and_visualize import test_model
preds, targs, attns = test_model(trained_model, test_loader)
```

---

## 🔮 Inference (via CLI)
```bash
python inference/predict.py 2 4 6 8 10
```
This prints the predicted next value and attention weights.

---

## 🔍 Example Predictions

| Input Sequence                        | Predicted Next Value | Attention Focus                     | Pattern Description             |
|--------------------------------------|-----------------------|-------------------------------------|---------------------------------|
| `[2.0, 4.0, 6.0, 8.0, 10.0]`          | **12.0786**           | `['0.002', '0.007', '0.084', '0.288', '0.618']` | Linear (+2)                    |
| `[3.0, 6.1, 9.2, 12.0]`              | **14.9393**           | `['0.001', '0.012', '0.175', '0.811']`         | Linear with noise              |
| `[0.0, 0.84, 0.91, 0.14, -0.76]`     | **-1.3314**           | `['0.020', '0.003', '0.019', '0.233', '0.725']` | Sinusoidal                    |
| `[4.39, 3.91, 3.58, 3.33, 3.14]`     | **3.2414**            | `['0.017', '0.249', '0.231', '0.208', '0.295']` | Logarithmic decay             |
| `[1.0, 2.0, 2.5, 2.0, 1.0]`          | **0.3634**            | `['0.012', '0.005', '0.015', '0.036', '0.932']` | Oscillatory/wave              |

> Attention highlights the most important timesteps the model focused on.

---

## 📝 Notes
- Use `weights/` for clean `.pt` weight files (state_dict only)
- Keep full PyTorch models in `saved_models/` if needed
- Use `models/` for reusable and clean architecture definitions
- Keep Jupyter experiments in `main.ipynb`, and modular scripts for training/inference

---
