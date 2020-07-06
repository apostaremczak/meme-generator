from dataclasses import dataclass


@dataclass
class MemeGeneratorConfig:
    max_seq_length: int = 1024
    vocab_size: int = 50304
    num_epochs: int = 8
