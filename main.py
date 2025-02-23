#main.py
from data_preparation import prepare_data
from model_building import build_model
from text_generation import generate_text
import numpy as np

# Example password dataset – in practice, use a larger dataset.
passwords = [
    "password", "123456", "qwerty", "abc123", "letmein", "monkey", "dragon" , "passwxyz1234", "Ronaldo5016!" , "##@@^^&&", " Password123!@#$%^&*()_+=-{}[]|:;<>,.?/~`" 
]

seq_length = 5
X, y, char_to_index, index_to_char, text = prepare_data(passwords, seq_length)
vocab_size = len(char_to_index)

model = build_model(vocab_size, seq_length)
# Train the model (adjust epochs and batch_size as needed).
model.fit(X, y, batch_size=16, epochs=100)

# Generate candidate password text
seed = "passw"  # Example seed; change as desired
generated_password = generate_text(model, seed, length=10, seq_length=seq_length,
                                   char_to_index=char_to_index, index_to_char=index_to_char,
                                   temperature=0.5)
print("Generated Password:", generated_password)