import nltk
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score

import ssl  # I don't know why, but I always have to run this bit anytime I'm first running NLTK

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


##### Take 2, in which I realize I, uh, didn't have to try doing all the CV by hand....


def make_features_dict(movie_reviews):
    # count vec and fit transform stuff
    movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
    bag_of_words = movie_vec.fit_transform(movie_reviews.data)
    return {
        "bag_of_words": bag_of_words,
        "tf_idf": TfidfTransformer().fit_transform(bag_of_words),
    }
    # going ahead and making tfidf version


def compare_metrics(bag_of_words_results_df, tf_idf_results_df):
    comparisons_list = []
    for metric in ("prec", "recall", "f1", "accuracy"):
        bag_of_words_best = bag_of_words_results_df[f"mean_test_{metric}"].max()
        tf_idf_best = tf_idf_results_df[f"mean_test_{metric}"].max()
        comparisons_list.append(
            (
                f"Performance according to {metric} metric (bag-of-words / TF-IDF): "
                f"{bag_of_words_best} / {tf_idf_best}"
            )
        )

    return "\n".join(comparisons_list)


from nltk.corpus import movie_reviews

# movie_reviews = load_files(movie_reviews.words(), shuffle=True)
features_dict = make_features_dict(movie_reviews)

# Set search specifications
scoring = {
    "prec": "precision",
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "accuracy": make_scorer(accuracy_score),
}
# clf = RandomizedSearchCV(
#     SVC(kernel="linear"),
#     distributions,
#     n_iter=12,
#     scoring=scoring,
#     cv=6,
#     return_train_score=True,
#     refit=False,
#     random_state=0,
#     n_jobs=12,
# )

scores_dict = {}
for feature_type, feature_dataset in features_dict.items():
    scores_dict[feature_type] = cross_validate(
        SVC(),
        feature_dataset,
        movie_reviews.target,
        cv=5,
        scoring={
            "prec": "precision",
            "recall": make_scorer(recall_score),
            "f1": make_scorer(f1_score),
            "accuracy": make_scorer(accuracy_score),
        },
        return_train_score=True,
        n_jobs=-1,
    )

print(compare_metrics(scores_dict["bag_of_words"], scores_dict["tf_idf"]))
