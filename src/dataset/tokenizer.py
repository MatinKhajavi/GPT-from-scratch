import numpy as np
from tiktoken import get_encoding
import torch

class Tokenizer:
    """
    Tokenizer class to handle the tokenization of documents using the tiktoken library for a specific encoding.
    
    :param encoding_name: The name of the encoding to use. Example: "gpt2"
    :type encoding_name: str
    """
    def __init__(self, encoding_name: str):
        self.enc = get_encoding(encoding_name)
        self.eot = self.enc._special_tokens['<|endoftext|>'] 


    def tokenize_doc(self, doc: dict, data_type: type = np.uint16) -> np.ndarray:
        """
        Tokenizes a single document and returns a numpy array of uint16 tokens.
        
        :param doc: A document with a 'text' field to tokenize.
        :type doc: dict
        :return: Tokenized document as a numpy array of uint16.
        :rtype: np.ndarray
        """
        tokens = [self.eot] 
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        
        return tokens_np.astype(data_type)


    def tokenize_str(self, string: str) -> torch.Tensor:
        """
        Tokenizes a single string and returns a tensor of integer tokens.

        :param string: A string to tokenize.
        :type string: str
        :return: Tokenized string as a tensor of integers.
        :rtype: torch.Tensor
        """

        encoded = self.enc.encode(string)
        return torch.tensor(encoded).unsqueeze(0)
    
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """
        Decodes a tensor of tokens into a string.
        
        :param tokens: A tensor of tokens to decode.
        :type tokens: torch.Tensor
        :return: A decoded string.
        :rtype: str
        """

        flat = tokens.squeeze(0)
        return self.enc.decode(flat.tolist())
        