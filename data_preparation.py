#data_preparation.py
import numpy as np

def prepare_data(passwords, seq_length, step=1):
    """
    Given a list of passwords and a sequence length, this function:
      - Combines the passwords into one text string (with newlines as separators)
      - Builds the character-to-index and index-to-character mappings
      - Creates input sequences (X) and their corresponding next-character outputs (y)
    Returns:
      X, y, char_to_index, index_to_char, text
    """
    text = "\n".join(passwords)
    chars = sorted(list(set(text)))
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for i, ch in enumerate(chars)}
    
    sentences = []
    next_chars = []
    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i:i+seq_length])
        next_chars.append(text[i+seq_length])
    
    X = np.zeros((len(sentences), seq_length), dtype=np.int32)
    y = np.zeros((len(sentences)), dtype=np.int32)
    
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t] = char_to_index[char]
        y[i] = char_to_index[next_chars[i]]
    
    return X, y, char_to_index, index_to_char, text