from dataclasses import dataclass


@dataclass
class MemeGeneratorConfig:
    max_seq_length: int = 512
