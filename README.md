# Semantle

Semantle is a game about guessing a secret word per the following rules:

This computer program attempts to compare user performance with a machine that auto-guesses the secret word using NLP techniques.

## User Guide

Semantle.py attempts to help the user solve the daily semantle game by by providing the user with words semantically similar to their input guesses in order to get closer to the target word in similarity score.

### Install Dependencies

To install all dependencies, run ```pip install -r requirements.txt```.

### Run File

Clone this github repo and run the main python file by typing python semantle.py in command prompt. <br>

Arguments: <br>
* --words: number of top similar words to print (defaults to 10)

EX ```python3 semantle.py --words 10```

Options:

* 'g': Enter guess words and the program will output top N semantically similar words to input list
* 'i': Semantic interpolation between a list of words given similarity scores list to target word. Output is a list of interpolated words.
* 'o1': Given a word, return words orthogonal to given word vector and therefore, most dissimilar.
* 'o2': Same as 'o1' option, with slower compute time and higher variance among output words. 

<hr>

## Rules

As cited from [Semantle](https://semantle.com/)
Each guess must be a word. Semantle will tell you how semantically similar it thinks your word is to the secret word. Unlike that other word game, it's not about the spelling; it's about the meaning. The similarity value comes from Word2vec. The highest possible similarity is 100 (indicating that the words are identical and you have won). The lowest in theory is -100, but in practice it's around -34. By "semantically similar", I mean, roughly "used in the context of similar words, in a database of news articles."

<hr>

## Techniques and Implementations

Every word in a Word2Vec model is a 700-dimensional vector embedding. By iteratively guessing words that are orthogonal to previous guesses in this hyperdimensional vector space, we maximize information gain with respect to the cosine similarity metric as a measure of word 'distance'. Recall that the cosine similarity between two vectors is defined as the cosine of the angle between them equal to the quotient of the their dot products and the product of their magnitudes. Based on the similarity score between the guess and target word, we can the guess at each timestep by finding a projection of the previous guess vector along a similar axis while exploring along a different dimension to maximize information gain until our guess is within a similarity delta of the target word. This is based on the principle and assumption that orthogonal word vectors have zero cosine similarity and are therefore semantically dissimilar, but contain semantic relevance along different concept dimensions. The most similar words to a given word are approximated using ANN and minimizing vector distance.

Interpolation <br>

* Given a list of vectors, V, and respective similarity scores as weights, W, the resultant averaged vector v' is approximated as follows:
$$
v' = 
\[
\sum_{i=1}^{n} v_i * w_i
\]
$$

Orthogonality <br>

* Given a word, a list of orthogonal word vectors contained in the plane normal to the given word is found according to the principle: C * V = 0, where C is the coefficient matrix for the hyperplane in standard form, multiplied with the components of V via dot product.
