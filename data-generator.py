# TODO: THAT IS NOT RELEVANT FILE.
#  FOR RELEVANT DATA GENERATION ADDRESS THE ./main.ipynb FILE

import torch
import numpy as np

def generate_linear_sequence(length, a, noise_std=0.0):
    x = np.random.uniform(1, 10)
    sequence = [x]
    for _ in range(length - 1):
        x = x + a + np.random.normal(0, noise_std)
        sequence.append(x)
    return torch.tensor(sequence, dtype=torch.float32)

def generate_logarithmic_sequence(length, a, b, c, noise_std=0.0):
    x = np.random.uniform(1, 3)
    sequence = [x]
    for _ in range(length - 1):
        x = a * np.log(x + b) + c + np.random.normal(0, noise_std)
        sequence.append(x)
    return torch.tensor(sequence, dtype=torch.float32)

def generate_exponential_sequence(length, a, noise_std=0.0):
    x = np.random.uniform(1, 3)
    sequence = [x]
    for _ in range(length - 1):
        x = a * np.exp(x) + np.random.normal(0, noise_std)
        sequence.append(x)
    return torch.tensor(sequence, dtype=torch.float32)

def generate_sinusoidal_sequence(length, A, omega, phi, noise_std=0.0):
    x = np.linspace(0, 2 * np.pi, length)
    sequence = A * np.sin(omega * x + phi) + np.random.normal(0, noise_std, size=length)
    return torch.tensor(sequence, dtype=torch.float32)

def generate_mixed_sequence(length, noise_std=0.0):
    seq1 = generate_linear_sequence(length // 2, a=2, noise_std=noise_std)
    seq2 = generate_exponential_sequence(length // 2, a=1.1, noise_std=noise_std)
    return torch.cat((seq1, seq2))

def generate_dataset(num_samples, length, max_length=20):
    data = []
    labels = []

    for _ in range(num_samples):
        seq_type = np.random.choice(['linear', 'quadratic', 'exponential', 'sinusoidal', 'mixed'])
        noise_std = np.random.uniform(0.0, 0.5)

        if seq_type == 'linear':
            sequence = generate_linear_sequence(length, a=np.random.uniform(1, 3), noise_std=noise_std)
        elif seq_type == 'logarithmic':
            sequence = generate_logarithmic_sequence(length, a=np.random.uniform(0.5, 1.5), b=np.random.uniform(1, 5), c=np.random.uniform(-2, 2), noise_std=noise_std)
        elif seq_type == 'exponential':
            sequence = generate_exponential_sequence(length, a=np.random.uniform(1.0, 1.5), noise_std=noise_std)
        elif seq_type == 'sinusoidal':
            sequence = generate_sinusoidal_sequence(length, A=np.random.uniform(1, 3), omega=np.random.uniform(0.5, 2.0), phi=np.random.uniform(0, 2 * np.pi), noise_std=noise_std)
        else:
            sequence = generate_mixed_sequence(length, noise_std=noise_std)

        sequence_len = np.random.randint(5, max_length + 1)
        sequence = sequence[:sequence_len]
        padding_len = max_length - sequence_len
        # Маска, где 1 — реальные данные, 0 — паддинг
        mask = [1] * sequence_len + [0] * padding_len
        # Заполнение нулями
        padded_sequence = torch.cat([sequence, torch.zeros(padding_len)])
        data.append((padded_sequence, torch.tensor(mask)))  # Входные данные - вся последовательность, кроме последнего элемента
        labels.append(sequence[-1])  # Последний элемент - это целевое значение

    return torch.stack(data), torch.tensor(labels)

if __name__ == "__main__":
    num_samples = 1_000_000
    seq_len = 20
    data, labels = generate_dataset(num_samples, seq_len)
    print(f"Data shape: {data.shape}")  # (1000, 19)
    print(f"Labels shape: {labels.shape}")  # (1000,)
