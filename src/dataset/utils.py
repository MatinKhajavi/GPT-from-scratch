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


def load_dataset(
        path: str,
        name: Optional[str] = None,
        split: Optional[Union[str, Split]] = None
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """
    Loads a dataset from the specified path, with optional specifications for dataset name and data split.

    This function wraps the Hugging Face datasets loading functionality, allowing users to specify the dataset they want to access by its path and, optionally, the specific configuration (name) and the particular split of the dataset (e.g., 'train', 'test').

    :param path: The path or name of the dataset. 
    :param name: Optional. Defining the name of the dataset configuration.
    :param split: Optional. Which split of the data to load. If `None`, will return a `dict` with all splits
    :return: Returns a dataset object or a dictionary of dataset objects depending on the splits specified. The type can be one of DatasetDict, Dataset, IterableDatasetDict, or IterableDataset.
    """
    return load_dataset(path=path, name=name, split=split)
