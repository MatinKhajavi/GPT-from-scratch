# GPT From Scratch

**GPT From Scratch** is an open-source implementation of a GPT-style transformer model, designed for scalability and efficiency. By leveraging Distributed Data Parallel (DDP) training, this project allows for efficient multi-GPU training, making it easier to train large-scale models on extensive datasets.

## Features

- Built from scratch GPT-style transformer model
- Efficient multi-GPU training using PyTorch's Distributed Data Parallel (DDP)
- Scalable and optimized for large datasets
- Easy-to-use interface for training and inference

## Results
Our model outperforms the GPT-2 checkpoint values on the HellaSwag benchmark, demonstrating the effectiveness of our implementation.

### Model Output
Model Before Training: 
```
On a clear day indis DNSヘラ ignore Happ Ce Croatian mugVAavorable303 wayomb prom bartender surmia pass standingotoshanMore intensely Lent loaf
```
Model After Training: 
```
On a clear day, our community is growing and we are enjoying the beauty and tranquility of our natural surroundings. While it may not be a
```

### Training Loss and HellaSwag Evaluation

Here is a figure showing the training/validation loss and model's performance on the HellaSwag benchmark:

![Model's Performance](results/fig.PNG)


## Installation

### Install from PyPI

To install the package from PyPI, run the following command:

```bash
pip install gpt_from_scratch
```

### Install from Github

To install the latest version directly from GitHub, use:

```bash
pip install git+https://github.com/MatinKhajavi/GPT-from-scratch.git
```

## Usage

### Tokenizing Data
Before training, you need to tokenize your data and prepare it in shards. Use the provided `tokenize_documents_parallel.py` script to process your data efficiently with multiprocessing.

**Command**:
```bash
python scripts/tokenize_documents_parallel.py --local_dir "edu_fineweb10B"
```

**Parameters**:
- `--local_dir`: Directory where tokenized data will be stored.
- `--remote_name`: Identifier for the remote dataset to download.
- `--shard_size`: Number of tokens each data shard will contain.
- `--dataset_path`: Full path to the dataset on the data hosting platform (e.g., Hugging Face).
- `--tokenizer_model`: The tokenizer model to use.

### Training the Model
You can train the model using a single GPU or multiple GPUs with Distributed Data Parallel (DDP).

#### Single GPU Training
**Command**:
```bash
python scripts/train_gpt.py
```

#### Multi-GPU Training
For multi-GPU training, ensure that your environment variables (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`) are set correctly to facilitate distributed training. The `torchrun` command simplifies this setup.

**Command**:
```bash
torchrun --standalone --nproc_per_node=8 scripts/train_gpt.py
```

**Parameters**:
- `--n_batches`: Number of batches for training or validation per iteration.
- `--n_tokens`: Number of tokens to process per batch.
- `--data_root`: Directory where tokenized data is stored.
- `--vocab_size`: Total number of unique tokens in the model's vocabulary.
- `--emb_dim`: Dimension of the embedding layer.
- `--context_length`: The length of the input sequences.
- `--drop_rate`: Dropout rate to use within the model to prevent overfitting.
- `--n_layers`: Number of layers in the transformer model.
- `--n_heads`: Number of attention heads in each transformer layer.
- `--qkv_bias`: Enable bias in the query, key, and value projections within attention layers.
- `--monitor`: Toggle to enable performance monitoring during training.
- `--torch_matmul_precision`: Precision setting for matrix multiplications in PyTorch.
- `--log_dir`: Directory to store training logs.
- `--n_epochs`: Total number of training epochs.
- `--warmup_iters`: Number of iterations to linearly increase the learning rate from zero to the initial rate.
- `--max_iters`: Maximum number of iterations to perform during training.
- `--total_batch_size`: Total batch size across all distributed training instances.
- `--metrics`: Metrics used to evaluate the model's performance.
- `--max_lr`: Maximum learning rate used in the learning rate scheduler.
- `--min_lr`: Minimum learning rate as part of the cyclical learning rate schedule.


### Example Commands
For a complete training session on 8 GPUs with specific parameters:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_gpt.py --data_root "data/edu_fineweb10B" --n_batches 16 --n_tokens 1024 --vocab_size 50304 --emb_dim 768 --context_length 1024 --n_layers 12 --n_heads 12
```


## License

This project is licensed under the MIT License.

