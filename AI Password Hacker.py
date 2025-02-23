import time
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense

# --- Step 1: Data Preparation ---
# A small dataset of example passwords (for demonstration only)
password_dataset = [
    "password", "123456", "qwerty", "abc123", "letmein", "monkey", "dragon" , "passwxyz1234", "Ronaldo5016!" , "##@@^^&&", " Password123!@#$%^&*()_+=-{}[]|:;<>,.?/~`" , "P@ssw0rd123" , "Passw0rd123!" , "Passw0rd123!!" , "P@ssw0rd"
]

# Combine passwords into one text (separated by newlines)
text = "\n".join(password_dataset)
# Create sorted list of unique characters
chars = sorted(list(set(text)))
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}

# Set sequence length for training sequences
seq_length = 5
sentences = []
next_chars = []
step = 1
for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i:i+seq_length])
    next_chars.append(text[i+seq_length])

# Convert sequences into numerical arrays
X = np.zeros((len(sentences), seq_length), dtype=np.int32)
y = np.zeros((len(sentences)), dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_to_index[char]
    y[i] = char_to_index[next_chars[i]]

vocab_size = len(chars)

# --- Step 2: Build and Train the Model ---
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

print("Training the neural network on sample passwords...")
model.fit(X, y, batch_size=16, epochs=100, verbose=1)

# --- Step 3: Define a Candidate Generator Using the Model ---
def generate_candidate(model, seed, target_length, seq_length, char_to_index, index_to_char, temperature=1.0):
    """Generate a candidate password from a seed using the trained model."""
    generated = seed
    while len(generated) < target_length:
        # Ensure we have a sequence of length seq_length; pad with spaces if needed
        seed_seq = generated[-seq_length:]
        if len(seed_seq) < seq_length:
            seed_seq = " " * (seq_length - len(seed_seq)) + seed_seq
        x_pred = np.zeros((1, seq_length), dtype=np.int32)
        for t, char in enumerate(seed_seq):
            x_pred[0, t] = char_to_index.get(char, 0)
        preds = model.predict(x_pred, verbose=0)[0]
        # Adjust predictions by temperature
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(range(vocab_size), p=preds)
        next_char = index_to_char[next_index]
        generated += next_char
    return generated

# --- Step 4: Define a Simple Password Strength Rating Function ---
def rate_password_strength(password):
    """Rates a password as Weak, Moderate, or Strong based on simple heuristics."""
    length = len(password)
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(c in string.punctuation for c in password)
    score = 0
    if length >= 8:
        score += 1
    if length >= 12:
        score += 1
    if has_upper:
        score += 1
    if has_digit:
        score += 1
    if has_symbol:
        score += 1

    if score <= 2:
        return "Weak"
    elif score <= 4:
        return "Moderate"
    else:
        return "Strong"

# --- Step 5: Brute-Force Candidate Generation with AI Guidance ---
target_password = input("Enter password to crack: ")
target_length = len(target_password)
# Use the first seq_length characters of the target (or the whole target if shorter) as the seed.
seed = target_password[:seq_length] if target_length >= seq_length else target_password

attempts = 0
found = False
start_time = time.time()

print("\nCracking password...")
while not found:
    candidate = generate_candidate(model, seed, target_length, seq_length, char_to_index, index_to_char, temperature=0.5)
    attempts += 1
    if candidate == target_password:
        found = True
        break

end_time = time.time()
elapsed = end_time - start_time

# --- Step 6: Output the Results ---
print("\nPassword found:", candidate)
print("Attempts:", attempts)
print("Time taken: {:.4f} seconds".format(elapsed))
print("Password strength rating:", rate_password_strength(target_password))
