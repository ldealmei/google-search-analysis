
# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import NMF, LatentDirichletAllocation

from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, OpenURL, Title, WheelZoomTool
from bokeh.layouts import row,column, widgetbox
from bokeh.models import FactorRange
from bokeh.models.widgets import Select
from bokeh.embed import components

from google_topic_utilities import *
from corpus_building import *
from utils import *

# -------------- Create corpus (Web scraping Google results) --------------
indir = ''
out_dir = ''

## Load query data
search_data = load_search_jsons(indir)

## Build corpus
search_data = search_data.drop_duplicates()
queries = search_data['query_text'].tolist()
print len(queries)

build_corpus(10, queries,outdir)

# -------------- Load, clean and prepare corpus data --------------
## Load data
indir = "Recherches/NLP/"
corpus_dict = load_corpus(indir)

## Process data
doc_clean = clean_text_data(corpus_dict)

## Create Document-Term Dataframe

tf_thresh = 10
tfidf_thresh = 0

doc_term_df = make_doc_term_df(doc_clean,'tfidf', tf_thresh, tfidf_thresh)
print "{} words from {} queries are included in the analysis".format(doc_term_df.shape[1],doc_term_df.shape[0])

# -------------- Apply Topic Detection Algorithm --------------
# Set Params
no_topics = 20
top_words = 10

# Run NMF
nmf = NMF(n_components=no_topics, random_state=42)
nmf.fit_transform(doc_term_df.values)

# Get all related matrices
V_df, W_df, H_df = get_NMF_df(doc_term_df,nmf)

# -------------- Display results --------------

# List
display_topics(nmf, doc_term_df.columns.tolist(), top_words)

# Word clouds
word_cloud_words = 100
dims = (1200,800) # Attention modif values in color_func
topics_wordcloud(nmf, doc_term_df.columns.tolist(), word_cloud_words, dims)

