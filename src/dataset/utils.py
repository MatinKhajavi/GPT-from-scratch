import numpy as np
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from datasets.arrow_dataset import Dataset
from datasets.splits import Split

def write_datafile(filename: str, tokens_np: np.ndarray) -> None:
    """
    Writes the tokenized data to a file.
    
    :param filename: The path to the file where the data will be saved.
    :param tokens_np: Numpy array of tokenized data to save.
    """
    np.save(filename, tokens_np)


def load_dataset(
        path: str,
        name: Optional[str] = None,
        split: Optional[Union[str, Split]] = None
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:

    return load_dataset("HuggingFaceFW/fineweb-edu", name=name, split=split)

