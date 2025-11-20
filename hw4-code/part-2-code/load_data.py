import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
# nltk.download('punkt')
nltk.download('punkt', quiet = True)
from transformers import T5Tokenizer, T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    # Dataset class for T5 text-to-SQL task
    # Loads natural language queries and their corresponding SQL queries

    def __init__(self, data_folder, split):
        
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # Initialize the T5Dataset
        # Args: 
            # data_folder: path to the data folder containing .nl and .sql files
            # split: one of 'train', 'dev', or 'test'
        self.split = split
        self.data_folder = data_folder

        # Debugging: validate data folder exists
        if not os.path.exists(data_folder):
            raise FileNotFoundError(
                f"Data folder not found: {data_folder}\n"
        )

        # Initialize tokenizer using t5-small (as specified)
        self.tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')

        # Load and process data
        self.data = self.process_data(data_folder, split, self.tokenizer)
        

    def process_data(self, data_folder, split, tokenizer):
        # Load and tokenize the natural language and SQL queries
        # Args: 
            # data_folder: path to data directory 
            # split: 'train', 'dev', or 'test'
            # tokenizer: T5 tokenizer
        # Returns: 
            # List of dictionaries containing tokenized inputs and targets
        
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        
        # Debugging: check if file exists
        if not os.path.exists(nl_path):
            raise FileNotFoundError(
                f"Natural language file not found: {nl_path}\n"
        )

        with open(nl_path, 'r', encoding='utf-8') as f: 
            nl_queries = [line.strip() for line in f.readlines()]  

        # Load SQL wqueries (not available for test set)
        sql_queries = None
        if split != 'test': 
            sql_path = os.path.join(data_folder, f'{split}.sql')

            # Debugging: check if file exists
            if not os.path.exists(sql_path):
                raise FileNotFoundError(
                f"SQL file not found: {sql_path}\n"
            )

            with open(sql_path, 'r', encoding = 'utf-8') as f: 
                sql_queries = [line.strip() for line in f.readlines()]
        
        # Process each example
        processed_data = []
        for idx, nl_query in enumerate(nl_queries): 
            # Tokenize input (natural language)
            # T5 expects task prefix -> e.g., "translate English to SQL: "
            input_text = f"translate English to SQL: {nl_query}"
            encoder_inputs = tokenizer(
                input_text, 
                add_special_tokens = True, 
                return_tensors = 'pt'
            )

            example = {
                'encoder_input_ids': encoder_inputs['input_ids'].squeeze(0), 
                'encoder_attention_mask': encoder_inputs['attention_mask'].squeeze(0), 
                'nl_query': nl_query # Keep original for debugging
            }

            # Tokenize target (SQL query) if available
            if sql_queries is not None: 
                sql_query = sql_queries[idx]
                # Tokenize SQL output
                decoder_outputs = tokenizer(
                    sql_query, 
                    add_special_tokens = True, 
                    return_tensors = 'pt'
                )
                example ['decoder_input_ids'] = decoder_outputs['input_ids'].squeeze(0)
                example['sql_query'] = sql_query # Keep original for debugging
            
            processed_data.append(example)

        return processed_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Extract encoder inputs
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_mask = [item['encoder_attention_mask'] for item in batch]
    decoder_ids = [item ['decoder_input_ids'] for item in batch]

    # Pad encoder sequences
    encoder_ids_padded = pad_sequence(
        encoder_ids, 
        batch_first = True, 
        padding_value = PAD_IDX
    )
    encoder_mask_padded = pad_sequence(
        encoder_mask, 
        batch_first = True, 
        padding_value = 0
    )

    # Pad decoder sequences
    decoder_ids_padded = pad_sequence(
        decoder_ids, 
        batch_first = True, 
        padding_value = PAD_IDX
    )

    # For T5, decoder_inputs are the tokens shifted right (all but last)
    # decoder_targets are the tokens shifted left (all but first)
    # However, T5 model handles this internally when we pass labels
    # We just need to prepare the full sequence

    # Decoder inputs: all tokens (T5 will handle teacher forcing internally)
    decoder_inputs = decoder_ids_padded

    # Decoder targets: same as decoder inputs (model will shift internally)
    # We need to mask padding tokens in loss computation
    decoder_targets = decoder_ids_padded.clone()

    # Initial decoder input for generation (just the start token)
    # For T5, we typically use pad_token_id as the start token
    # or the first token from decoder_ids
    initial_decoder_inputs = decoder_ids_padded[:, :1]  # First token (B x 1)

    return (
        encoder_ids_padded,      # (B, T)
        encoder_mask_padded,     # (B, T)
        decoder_inputs,          # (B, T')
        decoder_targets,         # (B, T')
        initial_decoder_inputs   # (B, 1)
    )
    # return [], [], [], [], []

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Extract encoder inputs
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_mask = [item['encoder_attention_mask'] for item in batch]
    
    # Pad encoder sequences
    encoder_ids_padded = pad_sequence(
        encoder_ids, 
        batch_first = True, 
        padding_value = PAD_IDX
    )
    
    encoder_mask_padded = pad_sequence(
        encoder_mask, 
        batch_first = True, 
        padding_value = 0
    )

    # For test set, we need initial decoder input
    # T5 typically uses pad_token_id as decoder_start_token_id
    batch_size = len(batch)
    # Get tokenizer to access pad_token_id
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    initial_decoder_inputs = torch.full(
        (batch_size, 1), 
        tokenizer.pad_token_id, 
        dtype = torch.long
    )
    
    return (
        encoder_ids_padded,      # (B, T)
        encoder_mask_padded,     # (B, T)
        initial_decoder_inputs   # (B, 1)
    )
    
    # return [], [], []

def get_dataloader(batch_size, split):
    # Create a DataLoader for the specified split
    # Args: 
        # batch_size: batch size for the dataloader
        # split: one of 'train', 'dev', or 'test'
    # Returns: 
        # DataLoader instance 
    
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        collate_fn = collate_fn
    )
    
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    # Load all data loaders for training, development, and test sets
    # Args: 
        # batch_size: batch size for training
        # test_batch_size: batch size for dev and test sets
    # Returns: 
        # Tuple of (train_loader, dev_loader, test_loader)

    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    # Helper function to load lines from a file
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load data for prompting experiments (optional, for LLM part of the assignment)
    # Returns: 
        # train_x, train_y, dev_x, dev_y, test_x
    
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x

if __name__ == "__main__":
    # Test the data loading
    print("Testing data loading...")

    # Load small batch
    train_loader = get_dataloader(batch_size = 4, split = "train")

    # Get one batch 
    for batch in train_loader:
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder = batch 
        print(f"Encoder IDs shape: {encoder_ids.shape}")
        print(f"Encoder mask shape: {encoder_mask.shape}")
        print(f"Decoder inputs shape: {decoder_inputs.shape}")
        print(f"Decoder targets shape: {decoder_targets.shape}")
        print(f"Initial decoder shape: {initial_decoder.shape}")
        break
    
    print("\nData loading test passed!")
