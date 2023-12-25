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
import pickle

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

def normalize(v):
    '''
    Normalize a vector to unit magnitude
    '''
    v_norm = v / np.linalg.norm(v)
    #v_norm = v_norm / np.linalg.norm(v_norm)
    return v_norm

def get_orthogonal(model, word, flag, max_return=10):
    '''
    Return N words from vectors orthogonal to word vector by finding projections on hyperplane
    '''
    v = get_vector(model, word)
    n = len(v)
    words = []
    valid = []

    # Create an identity matrix of size 300x300
    identity_matrix = np.eye(n)

    # Subtract the projection of v onto each standard basis vector
    # to obtain orthogonal vectors on the hyperplane perpendicular to v
    orthogonal_matrix = np.array([identity_matrix[i] - np.dot(v, identity_matrix[i]) / np.dot(v, v) * v for i in range(n)])

    # add valid words from orthogonal matrix
    count = 0
    for index, v in enumerate(orthogonal_matrix):
        if flag == '1' and count == max_return:
            break
        w = vec2word(model, [normalize(v)])
        if re.match("^[a-z]+$", w[0]):
            valid.append(w[0])
            #words.append(w)
            count += 1
    
    #orthogonal_vectors = orthogonal_matrix[:max_return]
    #normalized_vectors = [normalize(v) for v in orthogonal_vectors] # orthogonal_vectors / magnitude(orthogonal_vectors, axis=1)[:, np.newaxis]
    return np.random.choice(valid, max_return, replace=False) #words # normalized_vectors

def magnitude(vector):
    '''
    Compute magnitude of vector
    '''
    return np.linalg.norm(vector)

def interpolate(model, vocab, words, scores):
    vects = [get_vector(model, word) for word in words]
    n = len(vects[0])
    ave_vect = np.zeros(n)

    for i in range(n):
        for j in range(len(vects)):
            ave_vect[i] += scores[j] * vects[j][i]

    ave_vect = ave_vect / np.linalg.norm(ave_vect)
    ave_word = vec2word(model, [ave_vect])
    ave_words = skip_gram(model, vocab, ave_word, 10)
    
    return ave_vect, ave_words

###################################### MAIN ######################################

def main():
        
    #################### LOAD MODEL  #################
    # Load Google News Word2Vec Model

    model_path = './Word2Vec_Models/google_word2vec.pkl'

    # Load the Word2Vec model from the pickle file
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print('Model loaded')

    '''
    model_path = './Word2Vec_Models/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(model_path, limit=500000, binary=True)
    model.init_sims(replace=True) #Precompute L2-normalized vectors. If replace is set to TRUE, forget the original vectors and only keep the normalized ones. Saves lots of memory, but can't continue to train the model.
    '''
    
    # Filter out vectors not in the vocab list
    vocab = [word for word in model.index_to_key if re.match("^[a-zA-Z.-]+$", word)]
    
    #################### PARSE ARGS ###################

    parser = argparse.ArgumentParser(description="Semantic Analysis Script")
    parser.add_argument("--words", type=int, default=10, help="Maximum number of output words")
    #parser.add_argument("--max_words", type=int, default=10, help="Maximum number of words to consider in similarity")
    args = parser.parse_args()

    #################### USER INTERFACE #################

    msg = "\n\nWelcome to Semantle Solver. Command flags: \n \
           \'g\' for list of n similar words \n \
           \'i\' for word semantic interpolation \n \
           \'o12\' for orthogonal vectors most dissimilar to input word (use o1 for fast compute, o2 for slow higher variance) \n\n"
    print(msg)

    while True:
        
        guess, target = "woman", random.choice(vocab)

        # get guesses from user input

        mode = input("\nEnter command: ")
        guesses_input = input('\n\nEnter guess words (commas-separated): ')
        guesses = [guess.strip() for guess in guesses_input.split(',')]

        if mode == 'g':
            print("=== N Most Similar ===")
            top_n = n_most_similar_words(model, vocab, guesses, None, args.words)
            print_words(top_n)

        elif mode == 'i':
            score_input = input("Enter scores (comma-separated): ")
            scores = [float(score.strip()) for score in score_input.split(',')]
            vec, words = interpolate(model, vocab, guesses, scores)
            print("=== Interpolate ===")
            for w in words:
                print(w)
        
        elif mode == 'o1' or mode == 'o2':
            flag = mode[-1]
            print("=== Orthogonal Information Maximization ===")
            if len(guesses) > 1:
                    context_vectors = [model[word] for word in guesses]
                    avg_vector = np.mean(context_vectors, axis=0)
                    ave_word = vec2word(model, [avg_vector])
            else:
                ave_word = guesses[0]
            words = get_orthogonal(model, ave_word, flag)

            for w in words:
                print(w)
            '''
            try:
                if len(guesses) > 1:
                    context_vectors = [model[word] for word in guesses]
                    avg_vector = np.mean(context_vectors, axis=0)
                    ave_word = vec2word(model, [avg_vector])
                else:
                    ave_word = guesses[0]
                words = get_orthogonal(model, ave_word, flag)

                for w in words:
                    print(w)
            except:
                print("At least one word not in model vocab")
            '''

        else:
            print("Invalid Command Use: (g, i, o)")


if __name__ == '__main__':
    main()