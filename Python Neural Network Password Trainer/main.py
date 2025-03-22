# main.py
import pandas as pd
import numpy as np
import string
import time
from model_building import build_model
from text_generation import generate_candidate
from data_preparation import prepare_data

# --- Load password data ---
df = pd.read_csv("common_passwords.csv")  # Use full path if needed
passwords = df['password'].dropna().astype(str).tolist()
passwords = [pwd for pwd in passwords if len(pwd.strip()) >= 4]

# --- Prepare data for cracking ---
seq_length = 5
X, y, char_to_index, index_to_char, _ = prepare_data(passwords, seq_length)
vocab_size = len(char_to_index)

# --- Build & Train the model ---
model = build_model(vocab_size, seq_length)
model.fit(X, y, batch_size=16, epochs=100)

# --- Password strength rating ---
def rate_password_strength(password):
    length = len(password)
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(c in string.punctuation for c in password)
    score = 0
    if length >= 8: score += 1
    if length >= 12: score += 1
    if has_upper: score += 1
    if has_digit: score += 1
    if has_symbol: score += 1

    if score <= 2:
        return "Weak"
    elif score <= 4:
        return "Moderate"
    else:
        return "Strong"

# --- Cracking Simulation ---
target_password = input("Enter password to crack: ")
target_length = len(target_password)
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

elapsed = time.time() - start_time

# --- Results ---
print("\nPassword found:", candidate)
print("Attempts:", attempts)
print("Time taken: {:.4f} seconds".format(elapsed))
print("Password strength rating:", rate_password_strength(target_password))

# --- Save the trained model ---
model.save("password_cracker_model.h5")