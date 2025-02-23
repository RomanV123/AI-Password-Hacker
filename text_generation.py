# text_generation.py
import numpy as np

def generate_text(model, seed, length, seq_length, char_to_index, index_to_char, temperature=1.0):
    """
    Generates text using the trained model.
    Parameters:
      - model: the trained Keras model
      - seed: the initial text to start generating from
      - length: the number of characters to generate
      - seq_length: the sequence length used in training
      - char_to_index, index_to_char: mapping dictionaries
      - temperature: controls the randomness of predictions
    Returns:
      The generated text string.
    """
    generated = seed
    for _ in range(length):
        # Prepare input: pad or trim the seed to match seq_length.
        seed_seq = generated[-seq_length:]
        if len(seed_seq) < seq_length:
            seed_seq = " " * (seq_length - len(seed_seq)) + seed_seq
        x_pred = np.zeros((1, seq_length), dtype=np.int32)
        for t, char in enumerate(seed_seq):
            x_pred[0, t] = char_to_index.get(char, 0)
        
        preds = model.predict(x_pred, verbose=0)[0]
        # Apply temperature to the predictions
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        next_index = np.random.choice(len(preds), p=preds)
        next_char = index_to_char[next_index]
        generated += next_char
    return generated