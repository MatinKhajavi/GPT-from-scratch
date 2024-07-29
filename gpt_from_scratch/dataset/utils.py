import numpy as np
import torch
from tiktoken import get_encoding


def write_datafile(filename: str, tokens_np: np.ndarray) -> None:
    """
    Writes the tokenized data to a file.
    
    :param filename: The path to the file where the data will be saved.
    :param tokens_np: Numpy array of tokenized data to save.
    """
    np.save(filename, tokens_np)


def tokenize_str(string: str, encoding_name: str = "gpt2") -> torch.Tensor:
    """
    Tokenizes a single string and returns a tensor of integer tokens.

    :param string: A string to tokenize.
    :type string: str
    :param encoding_name: The name of the encoding to use. Example: "gpt2"
    :type encoding_name: str
    :return: Tokenized string as a tensor of integers.
    :rtype: torch.Tensor
    """
    enc = get_encoding(encoding_name)
    encoded = enc.encode(string)
    return torch.tensor(encoded).unsqueeze(0)


def decode_tokens(tokens: torch.Tensor, encoding_name: str = "gpt2") -> str:
    """
    Decodes a tensor of tokens into a string.
    
    :param tokens: A tensor of tokens to decode.
    :type tokens: torch.Tensor
    :param encoding_name: The name of the encoding to use. Example: "gpt2"
    :type encoding_name: str
    :return: A decoded string.
    :rtype: str
    """
    enc = get_encoding(encoding_name)
    flat = tokens.squeeze(0)
    return enc.decode(flat.tolist())