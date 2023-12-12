# Semantle

Semantle is a game about guessing a secret word per the following rules:

This computer program attempts to compare user performance with a machine that auto-guesses the secret word using NLP techniques.

## User Guide

Semantle.py attempts to help the user solve the daily semantle game by by providing the user with words semantically similar to their input guesses in order to get closer to the target word in similarity score.

### Install Dependencies

A requirements.txt will be added later. To install all dependencies, run pip install -r requirements.txt.

* pip install gensim
* pip install transformers
* pip install keyboard 

### Run File

Clone this github repo and run the main python file by typing python semantle.py in command prompt. <br>

Arguments: <br>
* --words: number of top similar words to print (defaults to 10)

python3 semantle.py --words 10

## Rules

As cited from [Semantle](https://semantle.com/)
Each guess must be a word. Semantle will tell you how semantically similar it thinks your word is to the secret word. Unlike that other word game, it's not about the spelling; it's about the meaning. The similarity value comes from Word2vec. The highest possible similarity is 100 (indicating that the words are identical and you have won). The lowest in theory is -100, but in practice it's around -34. By "semantically similar", I mean, roughly "used in the context of similar words, in a database of news articles."

## Techniques

Every word in a Word2Vec model is a 700-dimensional vector embedding. By iteratively guessing words that are orthogonal to previous guesses in this hyperdimensional vector space, we maximize information gain with respect to the cosine similarity metric. 
Recall that the cosine similarity between two vectors is defined as the cosine of the angle between them equal to the quotient of the their dot products and the product of their magnitudes. Based on the similarity score between the guess and target word, we update the guess at each timestep by finding a projection of the previous guess vector along a similar axis while exploring along a different dimension to maximize information gain until our guess is within a similarity delta of the target word.
