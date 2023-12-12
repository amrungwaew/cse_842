# import numpy as np
# import nltk as nl
import pandas as pd
from nltk.corpus import brown

## for some reason I sometimes get an error on my computer with nltk unless I run the ssl below, hence why it's there ##

# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

## MATRIX A (TRANSITION) ##


def make_markov_transition_matrix(tags_df):
    # making a new column to compare current tag with next tag
    tags_df["next_tag"] = tags_df.tag.shift(-1, fill_value=pd.NA)
    collapsed_tags_df = (
        tags_df.groupby(["tag", "next_tag"], as_index=False)
        .word.count()
        .rename(columns={"word": "num_obs"})
    )
    # calculating probability of each tag and saving
    collapsed_tags_df["prob"] = (collapsed_tags_df.num_obs) / (
        collapsed_tags_df.groupby("tag").num_obs.transform("sum")
    )
    # pivot for correct format
    return collapsed_tags_df.pivot(
        index="tag", columns="next_tag", values="prob"
    ).fillna(0)


brown_news_tagged = brown.tagged_words(categories="news", tagset="universal")
# making initial df with words and tags
tags_df = (
    pd.DataFrame([(word, tag) for word, tag in brown_news_tagged])
    .rename(columns={0: "word", 1: "tag"})
    .astype("string")
)

markov_transition_matrix = make_markov_transition_matrix(tags_df)
# print(markov_transition_matrix)

## MATRIX B (EMISSION) ##

# selection of words to filter by
select_words = ["science", "all", "well", "like", "but", "unlike", "today"]


def make_small_emission_matrix(tags_df, select_words):
    # making df and adding num_obs column to count word-tag pairs
    collapsed_tags_df = (
        tags_df.assign(num_obs=1)
        .groupby(["tag", "word"], as_index=False)
        .num_obs.count()
    )
    # calculating probabilities
    collapsed_tags_df["prob"] = (collapsed_tags_df.num_obs) / (
        collapsed_tags_df.groupby("tag").num_obs.transform("sum")
    )
    # filtering full probability matrix to only select the subset of choice words
    collapsed_tags_df = collapsed_tags_df.loc[
        collapsed_tags_df["word"].isin(select_words)
    ]
    # pivot for correct format
    return collapsed_tags_df.pivot(index="tag", columns="word", values="prob").fillna(0)


# to standardise/make the output more readable
pd.set_option("display.float_format", "{:.2g}".format)

small_emission_matrix = make_small_emission_matrix(tags_df, select_words)
# print(small_emission_matrix)
