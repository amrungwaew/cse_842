import pandas as pd
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from jax import random
import numpy as np
import os, glob
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
import nltk as nl


posi = []

path = "/Users/amrungwaew/Desktop/CSE 842/review_polarity/txt_sentoken/pos"  ## I know this is clunky but hey it works
for filename in glob.glob(os.path.join(path, "*.txt")):
    with open(os.path.join(os.getcwd(), filename), "r") as f:
        posi.append(f.read().replace("\n", ""))

neg = []

path = "/Users/amrungwaew/Desktop/CSE 842/review_polarity/txt_sentoken/neg"
for filename in glob.glob(os.path.join(path, "*.txt")):
    with open(os.path.join(os.getcwd(), filename), "r") as f:
        neg.append(f.read().replace("\n", ""))

dfpos = pd.DataFrame(posi)  # a df of all pos review text
dfpos["c"] = "positive"  # tagged class
dfneg = pd.DataFrame(neg)  # a df of all neg review text
dfneg["c"] = "negative"  # tagged class
# a df of all pos and neg, randomly shuffled
everything = (pd.concat([dfpos, dfneg])).sample(frac=1)

fold_1, fold_2, fold_3 = np.array_split(
    everything, 3
)  # creating 3 equal pieces for folds
# variables for fold 1
f1_X = fold_1.drop([fold_1.columns[-1]], axis=1)  # getting text only
f1_y = fold_1[fold_1.columns[-1]]  # getting labels only
# variables for fold 2
f2_X = fold_2.drop([fold_2.columns[-1]], axis=1)  # getting text only
f2_y = fold_2[fold_2.columns[-1]]  # getting labels only
# variables for fold 3
f3_X = fold_3.drop([fold_3.columns[-1]], axis=1)  # getting text only
f3_y = fold_3[fold_3.columns[-1]]  # getting labels only

# variables for fold 1 + fold 2 — fold 3 test
f12_X = pd.concat([f1_X, f2_X])
f12_y = pd.concat([f1_y, f2_y])
# variables for fold 1 + fold 3 — fold 2 test
f13_X = pd.concat([f1_X, f3_X])
f13_y = pd.concat([f1_y, f3_y])
# variables for fold 2 + fold 3 — fold 1 test
f23_X = pd.concat([f2_X, f3_X])
f23_y = pd.concat([f2_y, f3_y])


class Naive_Bayes(object):
    """in which I attempt to create a class for a NB classifier"""

    # splitting a given review (list with string) into word pieces
    def tokenize(self, text):
        for str_of_w in text:
            return str_of_w.split()

    # count how many times each word occurs given a list of words
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    # fitting everything...
    def fit(self, X, y):
        self.num_reviews = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()  # universal vocabulary
        n = len(X)  # number of reviews in X
        # counting the number of reviews in the respective categories
        self.num_reviews["positive"] = sum(1 for label in y if label == "positive")
        self.num_reviews["negative"] = sum(1 for label in y if label == "negative")
        # getting the prior probability based on the numbers in respective categories
        self.log_class_priors["positive"] = jnp.log(self.num_reviews["positive"] / n)
        self.log_class_priors["negative"] = jnp.log(self.num_reviews["negative"] / n)
        # counting how many words total in reviews in respective categories
        self.word_counts["positive"] = {}
        self.word_counts["negative"] = {}
        for x, y_1 in zip(X, y):
            c = "positive" if y_1 == "positive" else "negative"
            counts = self.get_word_counts(
                self.tokenize(x)
            )  # split review into word pieces
            for word, count in counts.items():  # for each word piece:
                if word not in self.vocab:
                    self.vocab.add(word)  # the universal vocabulary
                if word not in self.word_counts[c]:
                    self.word_counts[c][
                        word
                    ] = 0.0  # establishing counter in positive/negative count
                self.word_counts[c][word] += count

    # here goes nothing....
    def predict(self, X):
        result = []  # predicted label for each review
        for rev in X:  # review in list of reviews
            counts = self.get_word_counts(
                self.tokenize(rev)
            )  # for each review x, we tokenize into its pieces
            pos_probabilities = 0
            neg_probabilities = 0
            for word, w_count in counts.items():  # as counts is a dict
                if word not in self.vocab:
                    continue  # so it doesn't break
                # Calculating with +1 Laplace
                log_w_given_pos = jnp.log(
                    (self.word_counts["positive"].get(word, 0.0) + 1)
                    / (sum(self.word_counts["positive"].values()) + len(self.vocab))
                )
                log_w_given_neg = jnp.log(
                    (self.word_counts["negative"].get(word, 0.0) + 1)
                    / (sum(self.word_counts["negative"].values()) + len(self.vocab))
                )
                # accounting for the number of times a word appears
                log_w_pos_count = w_count * log_w_given_pos
                log_w_neg_count = w_count * log_w_given_neg
                # adding the probabilities to the sum
                pos_probabilities += log_w_pos_count  # summing the probabilities
                neg_probabilities += log_w_neg_count
            pos_probabilities += self.log_class_priors["positive"]  # adding the prior
            neg_probabilities += self.log_class_priors["negative"]
            if pos_probabilities > neg_probabilities:
                result.append("positive")
            else:
                result.append("negative")
        return result

    def accuracy_rate(self, result, actual):
        correct = 0
        for i in range(len(result)):
            if result[i] == actual[i]:
                correct += 1
        return (correct / float(len(result))) * 100.0


mod12 = Naive_Bayes()  # training on folds 1 and 2
mod23 = Naive_Bayes()  # training on folds 2 and 3
mod13 = Naive_Bayes()  # training on folds 1 and 3

f12X = (
    f12_X.values.tolist()
)  # have to convert these all to the correct form that will make NB happy
f12y = f12_y.tolist()

f13X = f13_X.values.tolist()
f13y = f13_y.tolist()

f23X = f23_X.values.tolist()
f23y = f23_y.tolist()

mod12.fit(f12X, f12y)
mod13.fit(f13X, f13y)
mod23.fit(f23X, f23y)

f3_pred = mod12.predict(f3_X.values.tolist())
f2_pred = mod13.predict(f2_X.values.tolist())
f1_pred = mod23.predict(f1_X.values.tolist())

mod12.accuracy_rate(f3_pred, f3_y.tolist())
# mod12.word_counts['negative']
# mod12.word_counts['positive']
# mod12.num_reviews
# mod12.vocab
# mod12.log_class_priors['negative']
# mod12.log_class_priors['positive']
# classification_report(f3_y.tolist(),f3_pred)

mod13.accuracy_rate(f2_pred, f2_y.tolist())
# mod13.word_counts['negative']
# mod13.word_counts['positive']
# mod13.vocab
# mod13.num_reviews
# mod13.log_class_priors['negative']
# mod13.log_class_priors['positive']
# classification_report(f2_y.tolist(),f2_pred)

mod23.accuracy_rate(f1_pred, f1_y.tolist())
# mod23.word_counts['negative']
# mod23.word_counts['positive']
# mod23.vocab
# mod23.num_reviews
# mod23.log_class_priors['negative']
# mod23.log_class_priors['positive']
# classification_report(f1_y.tolist(),f1_pred)
