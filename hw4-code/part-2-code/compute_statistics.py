"""
Script to compute data statistics before and after preprocessing for Q4.
This will help fill out Tables 1 and 2 in the assignment.
"""

import os
from transformers import T5TokenizerFast
from collections import Counter


def load_data(data_folder, split):
    # Load natural language and SQL queries
    nl_path = os.path.join(data_folder, f'{split}.nl')
    sql_path = os.path.join(data_folder, f'{split}.sql')
    
    with open(nl_path, 'r', encoding = 'utf-8') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    with open(sql_path, 'r', encoding = 'utf-8') as f:
        sql_queries = [line.strip() for line in f.readlines()]
    
    return nl_queries, sql_queries


def compute_vocabulary(queries):
    # Compute vocabulary size from a list of queries
    vocab = set()
    for query in queries:
        words = query.lower().split()
        vocab.update(words)
    return len(vocab)


def compute_statistics_before_preprocessing(data_folder):

    # Compute statistics before any preprocessing (Table 1).
    # Uses simple word-level tokenization.

    print("=" * 60)
    print("Table 1: Data Statistics BEFORE Preprocessing")
    print("=" * 60)
    
    for split in ['train', 'dev']:
        nl_queries, sql_queries = load_data(data_folder, split)
        
        num_examples = len(nl_queries)
        
        # Compute mean lengths (word-level)
        nl_lengths = [len(q.split()) for q in nl_queries]
        sql_lengths = [len(q.split()) for q in sql_queries]
        mean_nl_length = sum(nl_lengths) / len(nl_lengths)
        mean_sql_length = sum(sql_lengths) / len(sql_lengths)
        
        # Compute vocabulary sizes
        nl_vocab_size = compute_vocabulary(nl_queries)
        sql_vocab_size = compute_vocabulary(sql_queries)
        
        print(f"\n{split.upper()} SET:")
        print(f"  Number of examples: {num_examples}")
        print(f"  Mean NL sentence length (words): {mean_nl_length}")
        print(f"  Mean SQL query length (words): {mean_sql_length}")
        print(f"  Vocabulary size (natural language): {nl_vocab_size}")
        print(f"  Vocabulary size (SQL): {sql_vocab_size}")


def compute_statistics_after_preprocessing(data_folder):

    # Compute statistics after preprocessing (Table 2).
    # Uses T5 tokenizer for token-level statistics.

    print("\n" + "=" * 60)
    print("Table 2: Data Statistics AFTER Preprocessing")
    print("=" * 60)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    print(f"\nModel: google-t5/t5-small")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    for split in ['train', 'dev']:
        nl_queries, sql_queries = load_data(data_folder, split)
        
        # Tokenize with T5 tokenizer
        nl_tokenized = [tokenizer.encode(f"translate English to SQL: {q}") 
                       for q in nl_queries]
        sql_tokenized = [tokenizer.encode(q) for q in sql_queries]
        
        # Compute mean lengths (token-level)
        mean_nl_length = sum(len(tokens) for tokens in nl_tokenized) / len(nl_tokenized)
        mean_sql_length = sum(len(tokens) for tokens in sql_tokenized) / len(sql_tokenized)
        
        # Compute vocabulary sizes (unique tokens used)
        nl_vocab = set()
        sql_vocab = set()
        for tokens in nl_tokenized:
            nl_vocab.update(tokens)
        for tokens in sql_tokenized:
            sql_vocab.update(tokens)
        
        nl_vocab_size = len(nl_vocab)
        sql_vocab_size = len(sql_vocab)
        
        print(f"\n{split.upper()} SET:")
        print(f"  Mean NL sentence length (tokens): {mean_nl_length}")
        print(f"  Mean SQL query length (tokens): {mean_sql_length}")
        print(f"  Vocabulary size used (natural language): {nl_vocab_size}")
        print(f"  Vocabulary size used (SQL): {sql_vocab_size}")


def print_latex_table(data_folder):
    """
    Generate LaTeX-formatted tables for easy copy-paste into the report.
    """
    print("\n" + "=" * 60)
    print("LaTeX Formatted Tables")
    print("=" * 60)
    
    # Table 1
    print("\n% Table 1: Before Preprocessing")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Statistics Name & Train & Dev \\\\")
    print("\\hline")
    
    for split in ['train', 'dev']:
        nl_queries, sql_queries = load_data(data_folder, split)
        
        if split == 'train':
            num_examples = len(nl_queries)
            nl_lengths = [len(q.split()) for q in nl_queries]
            sql_lengths = [len(q.split()) for q in sql_queries]
            mean_nl_length = sum(nl_lengths) / len(nl_lengths)
            mean_sql_length = sum(sql_lengths) / len(sql_lengths)
            nl_vocab_size = compute_vocabulary(nl_queries)
            sql_vocab_size = compute_vocabulary(sql_queries)
            
            train_examples = num_examples
            train_nl = mean_nl_length
            train_sql = mean_sql_length
            train_nl_vocab = nl_vocab_size
            train_sql_vocab = sql_vocab_size
        
        else:
            num_examples = len(nl_queries)
            nl_lengths = [len(q.split()) for q in nl_queries]
            sql_lengths = [len(q.split()) for q in sql_queries]
            mean_nl_length = sum(nl_lengths) / len(nl_lengths)
            mean_sql_length = sum(sql_lengths) / len(sql_lengths)
            nl_vocab_size = compute_vocabulary(nl_queries)
            sql_vocab_size = compute_vocabulary(sql_queries)
            
            dev_examples = num_examples
            dev_nl = mean_nl_length
            dev_sql = mean_sql_length
            dev_nl_vocab = nl_vocab_size
            dev_sql_vocab = sql_vocab_size
    
    print(f"Number of examples & {train_examples} & {dev_examples} \\\\")
    print(f"Mean sentence length & {train_nl} & {dev_nl} \\\\")
    print(f"Mean SQL query length & {train_sql} & {dev_sql} \\\\")
    print(f"Vocabulary size (NL) & {train_nl_vocab} & {dev_nl_vocab} \\\\")
    print(f"Vocabulary size (SQL) & {train_sql_vocab} & {dev_sql_vocab} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Data statistics before any pre-processing.}")
    print("\\label{tab:stats_before}")
    print("\\end{table}")
    
    # Table 2
    print("\n% Table 2: After Preprocessing")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Statistics Name & Train & Dev \\\\")
    print("\\hline")
    print("\\multicolumn{3}{c}{\\textbf{Model: google-t5/t5-small}} \\\\")
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    for split in ['train', 'dev']:
        nl_queries, sql_queries = load_data(data_folder, split)
        
        nl_tokenized = [tokenizer.encode(f"translate English to SQL: {q}") 
                       for q in nl_queries]
        sql_tokenized = [tokenizer.encode(q) for q in sql_queries]
        
        mean_nl_length = sum(len(tokens) for tokens in nl_tokenized) / len(nl_tokenized)
        mean_sql_length = sum(len(tokens) for tokens in sql_tokenized) / len(sql_tokenized)
        
        nl_vocab = set()
        sql_vocab = set()
        for tokens in nl_tokenized:
            nl_vocab.update(tokens)
        for tokens in sql_tokenized:
            sql_vocab.update(tokens)
        
        if split == 'train':
            train_nl = mean_nl_length
            train_sql = mean_sql_length
            train_nl_vocab = len(nl_vocab)
            train_sql_vocab = len(sql_vocab)
        else:
            dev_nl = mean_nl_length
            dev_sql = mean_sql_length
            dev_nl_vocab = len(nl_vocab)
            dev_sql_vocab = len(sql_vocab)
    
    print(f"Mean sentence length (tokens) & {train_nl} & {dev_nl} \\\\")
    print(f"Mean SQL query length (tokens) & {train_sql} & {dev_sql} \\\\")
    print(f"Vocab size used (NL) & {train_nl_vocab} & {dev_nl_vocab} \\\\")
    print(f"Vocab size used (SQL) & {train_sql_vocab} & {dev_sql_vocab} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Data statistics after pre-processing with T5 tokenizer.}")
    print("\\label{tab:stats_after}")
    print("\\end{table}")


if __name__ == "__main__":
    data_folder = 'data'
    
    # Compute and print statistics
    compute_statistics_before_preprocessing(data_folder)
    compute_statistics_after_preprocessing(data_folder)
    print_latex_table(data_folder)
    
    print("\n" + "=" * 60)
    print("Statistics computation completed!")
    print("=" * 60)