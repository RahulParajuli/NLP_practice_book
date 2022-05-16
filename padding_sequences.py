from tokenize import Token
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'cat and dog are smart',
    'Do you really love my dog?',
    'there is a sheep in the dog cage',
]

tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding = 'post', truncating = 'post')

print("\n word index = ", word_index)
print("\n Sequences = ", sequences)
print ("\Padded sequences = ")
print(padded)
