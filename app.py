###################################### IMPORTS ######################################

from transformers import pipeline
from gensim.models import Word2Vec, Phrases
from gensim.models import KeyedVectors
import gensim.downloader
import random 
import numpy as np
import re
from scipy.spatial import distance
#simport keyboard
import argparse

from flask import Flask, render_template, request

'''
from views import views

'''

app = Flask(__name__)

###################################### HELPER FUNCTIONS ######################################

def get_vector(model, word):
    '''
    Model must be gensim object Word2Vec model
    Returns word vector if key present in its vocab
    '''
    try:
        vect = model[word]
    except:
        vect = None
        print("Word not in model vocabulary")
    return vect

def vec2word(model, vectors, top_n=1):
    '''
    Find most similar word in model given vector
    '''
    if isinstance(vectors, float):
        vectors = [vectors]

    # Find the word most similar to the given vector
    ave_vector = np.mean(vectors, axis=0)
    most_similar_words = model.similar_by_vector(ave_vector, topn=top_n+len(vectors))
    most_similar_words = [word for word, _ in most_similar_words]

    return most_similar_words[len(vectors):]

def cossim(model, vocab, word_1, word_2):
    # make sure both words in Word2Vec model
    if word_1 in vocab and word_2 in vocab:
        return (1 - distance.cosine(model[word_1], model[word_2])) 
    else:
        print("At least one word not in model vocab")
        return None
        
def n_most_similar_words(model, vocab, words, neg=None, n=10):
    '''
    negative is a list of words opposite of most similar n words
    '''

    if isinstance(words, str):
        words = [words]

    if (neg is None) and all(w in vocab for w in words):
        return model.most_similar(words, topn=n)
    elif (words is None) and all(n in vocab for n in neg):
        return model.most_similar(negative=neg, topn=n)
    elif all(w in vocab for w in words) and all(n in vocab for n in neg):
        return model.most_similar(positive=words, negative=neg, topn=n)
    else:
        print("Words not in model vocabulary")
        return None
    
def skip_gram(model, vocab, context_words, n):
    # Predict the most similar n words
    
    if isinstance(context_words, str):
        context_words = [context_words]

    if all(w in vocab for w in context_words):
        context_vectors = [model[word] for word in context_words]
        avg_vector = np.mean(context_vectors, axis=0)
        similar_words = model.similar_by_vector(avg_vector, topn=n+len(context_words))
        return similar_words[len(context_words):]

    else:
        print("Words not in model vocabulary")
        return None
    
def print_words(top_n):
    output = ""
    print('\n\n')
    for (word, sim) in top_n:
        output += f"{word} '\t\t' {sim}\n"
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    guess = request.form.get('guess')
    max_words = int(request.form.get('max_words'))

    # Load Google News Word2Vec Model
    model_path = './Word2Vec_Models/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(model_path, limit=500000, binary=True)
    model.init_sims(replace=True)

    # Filter out vectors not in the vocab list
    vocab = [word for word in model.index_to_key if re.match("^[a-zA-Z.-]+$", word)]

    target = random.choice(vocab)
    guesses = [guess] if guess else ['target', 'goal']  # get guesses from user input

    top_n = n_most_similar_words(model, vocab, guesses, None, max_words)
    output = print_words(top_n)

    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
