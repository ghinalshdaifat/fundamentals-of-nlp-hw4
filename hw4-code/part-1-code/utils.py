import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

'''
def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # Synonym Replacement Transformation
    # Replace approximately 25% of words with their synonyms from WordNet
    
    text = example["text"]
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Probability of replacing each word
    replacement_prob = 0.25
    
    transformed_tokens = []
    
    for token in tokens:
        # Only replace alphabetic tokens (skip punctuation, numbers)
        if token.isalpha() and random.random() < replacement_prob:
            # Get synonyms for this word
            synonyms = get_synonyms(token)
            
            if synonyms:
                # Replace with a random synonym
                transformed_tokens.append(random.choice(synonyms))
            else:
                # No synonyms found, keep original word
                transformed_tokens.append(token)
        else:
            # Don't replace this token
            transformed_tokens.append(token)
    
    # Detokenize back to text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tokens)
    
    ##### YOUR CODE ENDS HERE ######
    return example


def get_synonyms(word):
    """
    Get synonyms for a word using WordNet.
    Returns a list of synonyms (excluding the original word).
    """
    synonyms = set()
    
    # Get all synsets (synonym sets) for the word
    for syn in wordnet.synsets(word):
        # Get all lemmas (word forms) in each synset
        for lemma in syn.lemmas():
            synonym = lemma.name()
            
            # Replace underscores with spaces (WordNet uses underscores for phrases)
            synonym = synonym.replace('_', ' ')
            
            # Only add if it's different from the original word and is alphabetic
            if synonym.lower() != word.lower() and synonym.replace(' ', '').isalpha():
                synonyms.add(synonym)
    
    return list(synonyms)
'''
def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE ###
    
    # Random Typo Injection Transformation
    # Introduce realistic typos in approximately 15% of words
    
    text = example["text"]
    words = text.split()
    
    # Probability of introducing a typo in each word
    typo_prob = 0.50  # 50% of words
    
    transformed_words = []
    
    for word in words:
        # Only apply typos to alphabetic words longer than 3 characters
        if word.isalpha() and len(word) > 3 and random.random() < typo_prob:
            # Apply a random typo to this word
            word = apply_typo(word)
        
        transformed_words.append(word)
    
    example["text"] = ' '.join(transformed_words)
    
    ##### YOUR CODE ENDS HERE ######
    return example


def apply_typo(word):
    """
    Apply one random typo to a word.
    Typo types: character swap, deletion, insertion, or substitution.
    Uses QWERTY keyboard layout for realistic substitutions.
    """
    if len(word) < 2:
        return word
    
    # QWERTY keyboard neighbors for realistic typos
    keyboard_neighbors = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'sfcex', 'e': 'wsdr',
        'f': 'dgcvr', 'g': 'fhvbt', 'h': 'gjbny', 'i': 'ujko', 'j': 'hknmu',
        'k': 'jlomi', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx'
    }
    
    # Choose a random typo type
    typo_type = random.choice(['swap', 'delete', 'insert', 'substitute'])
    
    word_list = list(word.lower())
    pos = random.randint(0, len(word_list) - 1)
    
    if typo_type == 'swap' and len(word_list) > 2:
        # Swap two adjacent characters
        if pos < len(word_list) - 1:
            word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
    
    elif typo_type == 'delete' and len(word_list) > 2:
        # Delete a character
        word_list.pop(pos)
    
    elif typo_type == 'insert':
        # Insert a nearby key (keyboard neighbor)
        char = word_list[pos]
        if char in keyboard_neighbors:
            random_char = random.choice(keyboard_neighbors[char])
        else:
            random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        word_list.insert(pos, random_char)
    
    elif typo_type == 'substitute':
        # Substitute with a nearby key (keyboard neighbor)
        char = word_list[pos]
        if char in keyboard_neighbors:
            word_list[pos] = random.choice(keyboard_neighbors[char])
        else:
            # If not in keyboard map, use random letter
            word_list[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
    
    return ''.join(word_list)
