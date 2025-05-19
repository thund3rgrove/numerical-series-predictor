import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from multiprocessing import Pool

def generate_linear_sequence(length, a, noise_std=0.0, rng=None):
    rng = rng or np.random

    x = rng.uniform(1, 10)
    sequence = [x]
    for _ in range(length - 1):
        x = x + a + rng.normal(0, noise_std)
        sequence.append(x)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_logarithmic_sequence(length, a, b, c, noise_std=0.0, rng=None):
    rng = rng or np.random

    x = rng.uniform(1, 10)
    sequence = [x]
    for _ in range(length - 1):
        x = a * np.log(x + b) + c + rng.normal(0, noise_std)
        sequence.append(x)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_sinusoidal_sequence(length, A, omega, phi, noise_std=0.0, rng=None):
    rng = rng or np.random

    x = np.linspace(0, 2 * np.pi, length)
    sequence = A * np.sin(omega * x + phi) + rng.normal(0, noise_std, size=length)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_mixed_sequence(length, noise_std=0.0, rng=None):
    rng = rng or np.random

    half = length // 2
    seq1 = generate_linear_sequence(half, a=2, noise_std=noise_std, rng=rng)
    seq2 = generate_sinusoidal_sequence(length - half, A=1.5, omega=1.0, phi=0, noise_std=noise_std, rng=rng)
    return torch.cat((seq1, seq2))

def digit_sum(n):
    return sum(int(d) for d in str(abs(int(n))))

def generate_digit_sum_sequence(length, mutation_every=4, rng=None):
    rng = rng or np.random
    x = rng.randint(1, 10)
    sequence = [x]
    for i in range(1, length):
        if i % mutation_every == 0:
            src = sequence[i - mutation_every]
        else:
            src = sequence[-1]
        x = src + digit_sum(src)
        sequence.append(x)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_fibonacci_sequence(length, noise_std=0.0, rng=None):
    rng = rng or np.random
    a, b = rng.randint(1, 5), rng.randint(1, 5)
    sequence = [a, b]
    for _ in range(length - 2):
        x = sequence[-1] + sequence[-2] + rng.normal(0, noise_std)
        sequence.append(x)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_alternating_sequence(length, a=3.0, b=2.0, noise_std=0.0, rng=None):
    rng = rng or np.random
    x = rng.uniform(1, 5)
    sequence = [x]
    for i in range(1, length):
        if i % 2 == 1:
            x = x + a + rng.normal(0, noise_std)
        else:
            x = x - b + rng.normal(0, noise_std)
        sequence.append(x)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_modulo_sequence(length, mod=10, noise_std=0.0, rng=None):
    rng = rng or np.random
    x = rng.randint(mod, mod * 2)
    sequence = [x]
    for _ in range(length - 1):
        x = x % mod + rng.normal(0, noise_std)
        sequence.append(x)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_average_last_n_sequence(length, n=3, noise_std=0.0, rng=None):
    rng = rng or np.random
    sequence = [rng.uniform(1, 10) for _ in range(n)]
    for _ in range(length - n):
        avg = sum(sequence[-n:]) / n + rng.normal(0, noise_std)
        sequence.append(avg)

    sequence = torch.tensor(sequence, dtype=torch.float32)
    # sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
    return sequence

def generate_data(num_samples, max_length=20, seed=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)

    data, labels, masks = [], [], []

    # for _ in range(num_samples):
    for _ in trange(num_samples, desc='Generating sequences'):
        # seq_type = rng.choice(['linear', 'logarithmic', 'sinusoidal', 'mixed'])
        # seq_type = rng.choice([
        #     'linear', 'logarithmic', 'sinusoidal', 'mixed', 'digit_sum',
        #     'linear', 'logarithmic', 'sinusoidal', 'mixed',  # bias к обычным
        # ])
        # seq_type = rng.choice([
        #     'linear', 'logarithmic', 'sinusoidal', 'mixed', 'digit_sum',
        #     'fibonacci', 'alternating',
        #     'linear', 'logarithmic', 'sinusoidal', 'mixed'
        # ])

        # seq_type = rng.choice(
        #     ['linear', 'logarithmic', 'sinusoidal', 'mixed',
        #      'digit_sum', 'fibonacci', 'alternating', 'modulo', 'avg_last_n'],
        #     # p=[0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05]
        #     p=[
        #         0.18,  # linear
        #         0.18,  # logarithmic
        #         0.18,  # sinusoidal
        #         0.18,  # mixed
        #         0.08,  # digit_sum
        #         0.08,  # fibonacci
        #         0.05,  # alternating
        #         0.04,  # modulo
        #         0.03  # avg_last_n
        #     ]
        # )

        # NOTE: WORKING GOOD
        seq_type = rng.choice(['linear', 'logarithmic', 'sinusoidal', 'mixed'])
        
        # seq_type = rng.choice([
        #     'linear',        # 0.20
        #     'logarithmic',   # 0.20
        #     'sinusoidal',    # 0.20
        #     'mixed',         # 0.10
        #     'digit_sum',     # 0.10
        #     'fibonacci',     # 0.10
        #     'alternating',   # 0.05
        #     'modulo',        # 0.05
        # ], p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])


        noise_std = rng.uniform(0.0, 0.3)
        seq_len = rng.randint(5, max_length + 1)

        if seq_type == 'linear':
            sequence = generate_linear_sequence(seq_len + 1, a=rng.uniform(0.5, 2.0), noise_std=noise_std, rng=rng)
        elif seq_type == 'logarithmic':
            sequence = generate_logarithmic_sequence(seq_len + 1, a=rng.uniform(0.5, 1.5), b=1.0, c=0.5,
                                                     noise_std=noise_std, rng=rng)
        elif seq_type == 'sinusoidal':
            sequence = generate_sinusoidal_sequence(seq_len + 1, A=2.0, omega=rng.uniform(0.5, 2.0),
                                                    phi=rng.uniform(0, np.pi), noise_std=noise_std, rng=rng)
        elif seq_type == 'mixed':
            sequence = generate_mixed_sequence(seq_len + 1, noise_std=noise_std, rng=rng)
        elif seq_type == 'digit_sum':
            sequence = generate_digit_sum_sequence(seq_len + 1, rng=rng)
        elif seq_type == 'fibonacci':
            sequence = generate_fibonacci_sequence(seq_len + 1, noise_std=noise_std, rng=rng)
        elif seq_type == 'alternating':
            sequence = generate_alternating_sequence(seq_len + 1, noise_std=noise_std, rng=rng)
        elif seq_type == 'modulo':
            sequence = generate_modulo_sequence(seq_len + 1, mod=rng.randint(3, 12), noise_std=noise_std, rng=rng)
        elif seq_type == 'avg_last_n':
            sequence = generate_average_last_n_sequence(seq_len + 1, n=3, noise_std=noise_std, rng=rng)
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")

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
    data, labels, masks = generate_data(num_samples=1_000)  # or 10_000
    print("Data shape:", data.shape)  # (10000, 20)
    print("Labels shape:", labels.shape)  # (10000,)
    print("Masks shape:", masks.shape)  # (10000, 20)

    print('Sample data:')
    print(data[0])

