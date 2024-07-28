from dataclasses import dataclass



@dataclass
class GPTConfig:
    vocab_size: int = 50257  
    emb_dim: int = 768       
    context_length: int = 1024  
    drop_rate: float = 0.1 
    n_layers: int = 12      
    n_heads: int = 12       
    qkv_bias: bool = False  