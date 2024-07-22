import numpy as np
from tiktoken import get_encoding

class Tokenizer:
    """
    Tokenizer class to handle the tokenization of documents using the tiktoken library for a specific encoding.
    
    :param encoding_name: The name of the encoding to use. Example: "gpt2"
    :type encoding_name: str
    """
    def __init__(self, encoding_name: str):
        self.enc = get_encoding(encoding_name)
        self.eot = self.enc._special_tokens['<|endoftext|>'] 

    def tokenize(self, doc: dict, data_type: type = np.uint16) -> np.ndarray:
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
