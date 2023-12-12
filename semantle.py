###################################### IMPORTS ######################################

from transformers import pipeline
from gensim.models import Word2Vec, Phrases
from gensim.models import KeyedVectors
import gensim.downloader
import random 
import numpy as np
import re
from scipy.spatial import distance
import keyboard
import argparse

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
    print('\n\n')
    for (word, sim) in top_n:
        print(word, '\t\t', sim)

###################################### MAIN ######################################

def main():
        
    #################### LOAD MODEL  #################
    # Load Google News Word2Vec Model
    model_path = './Word2Vec_Models/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(model_path, limit=500000, binary=True)
    model.init_sims(replace=True) #Precompute L2-normalized vectors. If replace is set to TRUE, forget the original vectors and only keep the normalized ones. Saves lots of memory, but can't continue to train the model.
    #vocab = list(model.index_to_key)
    print('Model loaded')
    # filter model

    # Filter out vectors not in the vocab list
    vocab = [word for word in model.index_to_key if re.match("^[a-zA-Z.-]+$", word)]
    #model = KeyedVectors(vector_size=model.vector_size)
    #model.vocab = {word: model.vocab[word] for word in vocab if word in model.vocab}

    # If you want to keep the normalized vectors, you can initialize sims
    model.init_sims(replace=True)

    #################### PARSE ARGS ###################

    parser = argparse.ArgumentParser(description="Semantic Analysis Script")
    parser.add_argument("--words", type=int, default=10, help="Maximum number of output words")
    #parser.add_argument("--max_words", type=int, default=10, help="Maximum number of words to consider in similarity")
    args = parser.parse_args()

    #################### USER INTERFACE #################

    while True:
        
        guess, target = "woman", random.choice(vocab)

        # get guesses from user input

        guesses_input = input('\n\nEnter guess words (commas-separated): ')
        guesses = [guess.strip() for guess in guesses_input.split(',')]
        #['target', 'goal'] 

        #print(guess, target)

        top_n = n_most_similar_words(model, vocab, guesses, None, args.words)
        print_words(top_n)

if __name__ == '__main__':
    main()