import nltk
import pickle
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score
from scipy.stats import uniform
import random

# easier than big ugly paths everywhere
data_dir = "/Users/amrungwaew/Desktop/CSE 842"
working_dir = "/Users/amrungwaew/Desktop/CSE 842"

# loading data
movie_reviews = load_files(f"{data_dir}/movie_reviews", shuffle=True)

# count vec for BoW, tfidf
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
bag_of_words = movie_vec.fit_transform(movie_reviews.data)
features_dict = {
    "bag_of_words": bag_of_words,
    "tf_idf": TfidfTransformer().fit_transform(bag_of_words),
}

scoring = {
    "prec": "precision",
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "accuracy": make_scorer(accuracy_score),
}

#################################
########### PART 1 ###########
##################################

# making MNB, cp = None model
mnb_N = MultinomialNB(class_prior=None)

# making MNB, cp = [1,2] model

######################################

results_bow = []
results_tfidf = []

random.seed(10)

for feature_type, feature_dataset in features_dict.items():
    mnb_N_results = cross_validate(
        mnb_N,
        X=feature_dataset,
        y=movie_reviews.target,
        cv=5,
        n_jobs=9,
        scoring=scoring,
    )
    mnb_N_df = pd.DataFrame(mnb_N_results)

    if feature_type == "bag_of_words":
        for key in mnb_N_results.keys():  # since the MNB scoring outputs are funky
            if key in ["fit_time", "score_time"]:
                pass  # omitting these to make it run a bit faster/don't need them
            else:
                mean = mnb_N_df[f"{key}"].mean()  # the mean score
                results_bow.append(
                    f"MNB_N performance {key} (bag-of-words): " f"{mean}\n"
                )

    elif feature_type == "tf_idf":
        for key in mnb_N_results.keys():
            if key in ["fit_time", "score_time"]:
                pass
            else:
                mean = mnb_N_df[f"{key}"].mean()
                results_tfidf.append(f"MNB_N performance {key} (TF-IDF): " f"{mean}\n")


print("\n".join(results_bow))
print("\n".join(results_tfidf))

##################################
######## PART 2 ################
######################################

# Set search specifications
distributions = {"C": uniform(loc=0, scale=100)}  # search space
# metrics

random.seed(11)

# making svc model
svc_clf = RandomizedSearchCV(
    SVC(kernel="linear"),
    distributions,
    n_iter=9,
    scoring=scoring,
    cv=4,
    return_train_score=True,
    refit=False,
    random_state=0,
    n_jobs=9,
)


# going thru two item dict of BoW, tfidf
for feature_type, feature_dataset in features_dict.items():
    svc_search_results = svc_clf.fit(feature_dataset, movie_reviews.target)
    with open(f"{working_dir}/{feature_type}_svc_results.pkl", "wb") as handle:
        pickle.dump(svc_search_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


######################################

# Load BoW results SVC
with open(
    "/Users/amrungwaew/Desktop/CSE 842/bag_of_words_svc_results.pkl", "rb"
) as handle:
    # opening again
    svc_bag_of_words_results = pickle.load(handle)
svc_bag_of_words_results_df = pd.DataFrame(svc_bag_of_words_results.cv_results_)

# Load tfidf results SVC
with open("/Users/amrungwaew/Desktop/CSE 842/tf_idf_svc_results.pkl", "rb") as handle:
    # opening again
    svc_tf_idf_results = pickle.load(handle)
svc_tf_idf_results_df = pd.DataFrame(svc_tf_idf_results.cv_results_)


#####################

comparisons_list = []
for metric in ("prec", "recall", "f1", "accuracy"):
    #### SVC

    svc_bag_of_words_best = svc_bag_of_words_results_df[
        f"mean_test_{metric}"
    ].max()  # getting the best one
    svc_tf_idf_best = svc_tf_idf_results_df[
        f"mean_test_{metric}"
    ].max()  # getting the best one

    comparisons_list.append(
        (
            #### SVC
            f"SVC performance {metric} metric (bag-of-words): "
            f"{svc_bag_of_words_best}\n"
            f"SVC performance {metric} metric (tf-idf): "
            f"{svc_tf_idf_best}"
        )
    )

# print("\n".join(comparisons_list))
