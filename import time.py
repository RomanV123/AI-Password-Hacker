import time
import itertools

password = input('Enter Password: ')
start = time.time()

letters = 'abcdefghijklmnopqrstuvwxyz'
numbers = '0123456789'
symbols = '!@#$%^&*()'
chars = letters + numbers + symbols

found = False

# Try all combinations for lengths 1 to 5 (adjust max length as needed)
for length in range(1, 6):
    for candidate in itertools.product(chars, repeat=length):
        candidate_str = ''.join(candidate)
        if candidate_str == password:
            found = True
            break
    if found:
        break

end = time.time()
clock = end - start

print('Your password: ' + password)
print('Time taken: ' + str(clock))
