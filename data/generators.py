import torch
import torch.nn.functional as F
import numpy as np

def generate_linear_sequence(length, a, noise_std=0.0, rng=None):
    rng = rng or np.random

    x = rng.uniform(1, 10)
    sequence = [x]
    for _ in range(length - 1):
        x = x + a + rng.normal(0, noise_std)
        sequence.append(x)
    return torch.tensor(sequence, dtype=torch.float32)

def generate_logarithmic_sequence(length, a, b, c, noise_std=0.0, rng=None):
    rng = rng or np.random

    x = rng.uniform(1, 10)
    sequence = [x]
    for _ in range(length - 1):
        x = a * np.log(x + b) + c + rng.normal(0, noise_std)
        sequence.append(x)
    return torch.tensor(sequence, dtype=torch.float32)

def generate_sinusoidal_sequence(length, A, omega, phi, noise_std=0.0, rng=None):
    rng = rng or np.random

    x = np.linspace(0, 2 * np.pi, length)
    sequence = A * np.sin(omega * x + phi) + rng.normal(0, noise_std, size=length)
    return torch.tensor(sequence, dtype=torch.float32)

def generate_mixed_sequence(length, noise_std=0.0, rng=None):
    rng = rng or np.random

    half = length // 2
    seq1 = generate_linear_sequence(half, a=2, noise_std=noise_std, rng=rng)
    seq2 = generate_sinusoidal_sequence(length - half, A=1.5, omega=1.0, phi=0, noise_std=noise_std, rng=rng)
    return torch.cat((seq1, seq2))

def generate_dataset(num_samples, max_length=20, seed=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)

    data = []
    labels = []
    masks = []

    for _ in range(num_samples):
        seq_type = rng.choice(['linear', 'logarithmic', 'sinusoidal', 'mixed'])
        noise_std = rng.uniform(0.0, 0.3)
        seq_len = rng.randint(5, max_length + 1)

        if seq_type == 'linear':
            sequence = generate_linear_sequence(seq_len + 1, a=rng.uniform(0.5, 2.0), noise_std=noise_std, rng=rng)
        elif seq_type == 'logarithmic':
            sequence = generate_logarithmic_sequence(seq_len + 1, a=rng.uniform(0.5, 1.5), b=1.0, c=0.5, noise_std=noise_std, rng=rng)
        elif seq_type == 'sinusoidal':
            sequence = generate_sinusoidal_sequence(seq_len + 1, A=2.0, omega=rng.uniform(0.5, 2.0), phi=rng.uniform(0, np.pi), noise_std=noise_std, rng=rng)
        else:
            sequence = generate_mixed_sequence(seq_len + 1, noise_std=noise_std, rng=rng)

        x = sequence[:-1]  # input sequence
        y = sequence[-1]   # target next value
        padding = max_length - seq_len

        padded_x = F.pad(x, (0, padding), value=0.0)
        mask = torch.tensor([1] * seq_len + [0] * padding, dtype=torch.bool)

        data.append(padded_x)
        labels.append(y)
        masks.append(mask)

    return torch.stack(data), torch.stack(labels), torch.stack(masks)


if __name__ == '__main__':
    data, labels, masks = generate_dataset(num_samples=1_000)  # or 10_000
    print("Data shape:", data.shape)  # (10000, 20)
    print("Labels shape:", labels.shape)  # (10000,)
    print("Masks shape:", masks.shape)  # (10000, 20)

    print('Sample data:')
    print(data[0])

