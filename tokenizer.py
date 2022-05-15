from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [
    'I love my dog',
    'I, love my cat',
    'Do you love my dog?',
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequence = tokenizer.texts_to_sequences(sentences)
print(word_index)
print(sequence)

