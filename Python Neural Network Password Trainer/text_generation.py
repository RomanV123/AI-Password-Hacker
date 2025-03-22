# text_generation.py
import numpy as np

def generate_candidate(model, seed, target_length, seq_length, char_to_index, index_to_char, temperature=1.0):
    generated = seed
    while len(generated) < target_length:
        seed_seq = generated[-seq_length:]
        if len(seed_seq) < seq_length:
            seed_seq = " " * (seq_length - len(seed_seq)) + seed_seq
        x_pred = np.zeros((1, seq_length), dtype=np.int32)
        for t, char in enumerate(seed_seq):
            x_pred[0, t] = char_to_index.get(char, 0)
        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(len(preds), p=preds)
        next_char = index_to_char[next_index]
        generated += next_char
    return generated
