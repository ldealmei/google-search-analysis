# coding: utf-8
import numpy as np
import pandas as pd
import json
import datetime
import io
import os
import re
from tqdm import tqdm

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from sklearn.decomposition import PCA


# Load data
def load_corpus(indir,test_lim = np.Inf) :
    docs = []
    c = 0 
    for filename in os.listdir(indir):
        if c > test_lim :
            break
        if filename.endswith('.txt'):
            f_ = io.open(os.path.join(indir,filename), mode="r", encoding="utf-8")
            txt = f_.read()
            docs.append(json.loads(txt))
            c +=1

    # flatten docs
    docs = [sublist2 for sublist1 in docs for sublist2 in sublist1]
    
    return {k : doc for i in range(len(docs)) for k,doc in docs[i].items()}


def clean(doc, stop, exclude, lemma):
    
    # Delete annoying [l',d',qu',...]
    doc_clean = " ".join([w if not re.match("^l'|^d'|^qu'",w) else re.sub("^l'|^d'|^qu'",'',w) for w in doc.lower().split() ])
    
    # Delete stop words
    stop_free = " ".join([i for i in doc_clean.split() if i not in stop])

    # Delete punctuation
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # Apply lemmatisation
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    # Get rid of numbers
    clean = [w for w in normalized.split() if not re.search('^[0-9]*$',w)]
    
    # Get rid of particular characters
#     clean = [w for w in clean if not re.search('·',w)]

    return clean

def clean_text_data(corpus_dict) :
    
    stop_en = stopwords.words('english')
    #     stop_fr = set(stopwords.words('french'))
    stop_nl = stopwords.words('dutch')
    
    # Load a more complete stop word list from https://github.com/stopwords-iso/stopwords-fr
    with open('stopwords-fr.json','r') as f :
        stop_fr = json.load(f)

    #     stop_custom = [u"l'",u"d'",u"qu'"]
    stop = set(stop_en + stop_fr + stop_nl)

base_excl = set(string.punctuation)
    cust_excl = set([u'\xb7',u'\xab',u'\xbb',u'\u30fb',u'\uff09',u'\u2014',u'\u2713'])
    exclude = base_excl.union(cust_excl)
    lemma = WordNetLemmatizer()
    
    doc_clean = {k : clean(" ".join([d for d in doc]), stop, exclude, lemma) for k,doc in corpus_dict.items()}
    
    return doc_clean

def make_doc_term_df(corpus, method = 'tf', tf_thresh = 0, tfidf_thresh = 0) :
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(corpus.values())

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = {k : dict(dictionary.doc2bow(doc)) for k,doc in corpus.items()}

    # Create doc-term dataframe
    nb_words = len(dictionary.keys())
    dict_df = {q : [doc_term_matrix[q].get(i,0) for i in range(nb_words)] for q in corpus.keys()}
    doc_term_df = pd.DataFrame(dict_df).T
    sort_idx = np.array(dictionary.keys()).argsort()
    doc_term_df.columns = np.array(dictionary.values())[sort_idx]
    
    # Apply threshold
    doc_term_df = doc_term_df.loc[:,(doc_term_df.sum() > tf_thresh)]
    
    # Doc-term but with tf-idf
    ## idf for each term 
    idf_s = -np.log((doc_term_df>0).sum()/doc_term_df.shape[1])
    # Modify matrix values
    doc_tfidf_df = doc_term_df.apply(lambda s : s*idf_s.loc[s.name] )    
    doc_tfidf_df = doc_tfidf_df.loc[:,(doc_tfidf_df.max() > tfidf_thresh)]
    
    if method == 'tf' :
        return doc_term_df
    elif method == 'tfidf':
        return doc_tfidf_df

def display_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def display_topics_val(model, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([str(np.round(topic[i],2))
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def display_topics_by_q(model, feature_names, q):
    for topic_idx, topic in enumerate(model.components_):
        subset = topic > np.percentile(topic,q)
        message = "Topic #%d: " % topic_idx
        message += " ".join([np.array(feature_names)[subset][i] for i in topic[subset].argsort()[::-1]])
        print(message)
    print()
    
# ----------------------- TOPIC WORDCLOUDS -----------------------

def get_topic_wordcloud(topic_idx, model, feature_names, n_top_words):
    
    topic = model.components_[topic_idx]
    cloud_data ={feature_names[i]: np.round(topic[i],2) for i in topic.argsort()[:-n_top_words - 1:-1]}
        
    wordcloud = WordCloud(background_color = 'white', 
                          color_func = color_words).generate_from_frequencies(cloud_data)
    
    ax = plt.imshow(wordcloud, interpolation=None)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title("Topic #%d " % topic_idx)
    plt.imsave('topic%d.jpg' % topic_idx,ax.get_array())
    return ax
    
def color_words(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None) :
    # MAY CHANGE, max_font_size changes with image height !! CAREFUL
    min_font_size = 4
    max_font_size = 800
    
    font_range = max_font_size - min_font_size
    
    tier1 = (min_font_size, min_font_size+ np.round(0.15*font_range,0))
    tier2 = (min_font_size+ np.round(0.15*font_range,0) + 1, min_font_size+ np.round(0.3*font_range,0))
    tier3 = ( min_font_size+ np.round(0.3*font_range,0)+1, max_font_size)

    tiers = [tier1, tier2, tier3]
    palette = ["#008DCB", "#F47D4A", "#E1315B"]

    tier = [i for i in range(len(tiers)) if (font_size >= tiers[i][0] and font_size <= tiers[i][1])]
    if not tier :
        print font_size
    PIL_color = palette[tier[0]]
    
    return PIL_color

def topics_wordcloud(model, feature_names, n_top_words, dims):
    """ Shows all wordclouds and saves them to a file"""
    plt.figure(figsize = (20,20))
    for topic_idx, topic in enumerate(model.components_):
        cloud_data ={feature_names[i]: np.round(topic[i],2) for i in topic.argsort()[:-n_top_words - 1:-1] if np.round(topic[i],2) != 0.0 }
        wordcloud = WordCloud(background_color = 'white', 
                              color_func = color_words,
                             width = dims[0],
                             height = dims[1])
        wordcloud = wordcloud.generate_from_frequencies(cloud_data)
        plt.subplot(5,4,topic_idx+1)
        ax = plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.title("Topic #%d " % topic_idx)
        
        
        plt.imsave('topic%d.jpg' % topic_idx,ax.get_array())

    plt.show()
# -------------------------------------------------------------------------------------

def get_NMF_df(doc_term_df,nmf_model) :
    """ Get all the matrices of the NM factorization as a DataFrame. Matrice notation follow those of Wikipedia for NMF"""
    
    V = doc_term_df.values
    
    # Get H matrice and build DataFrame
    H = nmf_model.components_
    H_df = pd.DataFrame(H, columns = doc_term_df.columns, index=["Topic #%d" % i for i in range(H.shape[0])])
    
    # Compute W matrice and build DataFrame
    H_inv = H.transpose()
    W = np.dot(V, H_inv)
    W_df = pd.DataFrame(W, index = doc_term_df.index, columns=["Topic #%d" % i for i in range(H_inv.shape[1])])
    
    return doc_term_df, W_df, H_df

# -------------------------------------------------------------------------------------
def format_topic_for_display(W_df, H_df, base_url, no_topics) :
    """Prepare dataframe for displaying in bokeh plot"""
    doc_topic = W_df.idxmax(axis=1)
    docs_by_topic = [len(doc_topic[doc_topic == topic]) for topic in doc_topic.unique()]
    pca = PCA(n_components =2, random_state =42 )
    pca.fit(H_df.values)
    print pca.components_.shape
    print pca.explained_variance_ratio_
    
    topics_2d = pd.DataFrame(pca.transform(H_df.values), index = H_df.index, columns = ['PC1','PC2'])
    topics_2d['size'] = docs_by_topic
    topics_2d['viz_size'] = np.sqrt(docs_by_topic)
    topics_2d['url'] = [base_url + 'topic%d.jpg' % i for i in range(no_topics)]
    
    return topics_2d
# -------------------------------------------------------------------------------------

def multiple_check(lst) :
    if len(lst) == 1 :
        return 0
    else :
        return 1
    
def get_time(lst):
    # We discard the multiple timestamps
    return datetime.datetime.fromtimestamp(int(lst[0]['timestamp_usec'])/1e6)

def load_search_jsons(indir) :
    queries_data = pd.DataFrame()
    
    for filename in os.listdir(indir):
        if filename.endswith('.json'):
            with open(os.path.join(indir,filename),'r') as f_ :
                _json = json.load(f_)
            
            _data = pd.DataFrame([i['query'] for i in _json['event']])
            queries_data = pd.concat([queries_data, _data])
    
    queries_data['time'] = queries_data['id'].apply(get_time)
    queries_data['multiple_timestamps'] = queries_data['id'].apply(multiple_check)
    del queries_data['id']
    
    queries_data = queries_data.set_index('time') 
    
    return queries_data
# -------------------------------------------------------------------------------------

def q_list(s):
    return [q for q in s['query_text']]

def get_topic_score(bow_idx, topic) :
    
    score = np.sum(topic[bow_idx])
    return score

def find_topic(bow, model_bow, model) :
    #Get indices of thw words in the matrix
    common_bow_idx = [model_bow.index(w) for w in bow if w in model_bow]
    #     print "    " + "{:.2%} of bow conserved".format(float(len(common_bow_idx))/len(bow))
    scores = [get_topic_score(common_bow_idx, topic) for topic in model.components_]
    return np.argmax(scores), np.max(scores)

def get_query_topic_df(doc_clean, search_data,doc_term_df, no_topics, nmf_model) :
    searches_by_date = search_data.groupby(by = search_data.index.date).apply(q_list)
    
    search_topics_df = pd.DataFrame(columns = ['Term'] + ["Topic #%d" % i for i in range(no_topics)])
    
    for i in range(searches_by_date.shape[0]) :
        new_df = pd.DataFrame(searches_by_date[i], columns = ['Term'], index = [searches_by_date.index[i]]*len(searches_by_date[i]))
        search_topics_df = pd.concat([search_topics_df,new_df])
    
    search_topics_df = search_topics_df.fillna(value = 0)
    
    doc_list = {search : doc_clean[search] for search in search_topics_df['Term'].unique() if search in doc_clean.keys()}


    for term, bow in tqdm(doc_list.items()) :
        sel = search_topics_df['Term'] == term
        topic_nr, score = find_topic(bow,doc_term_df.columns.tolist(), nmf_model)
        search_topics_df.loc[sel,'Topic #%d' % topic_nr] = 1
        search_topics_df.loc[sel,'Score'] = score
    
    search_topics_df.index = pd.to_datetime(search_topics_df.index)
    
    return search_topics_df

def add_example_search_terms(search_df, month, year, topic_nr, nb_terms) :
    
    cond_date = np.logical_and(search_df.index.month == month,search_df.index.year == year)
    cond_topic = search_df['Topic #%d' % topic_nr]==1
    interm = search_df.loc[np.logical_and(cond_date,cond_topic), ['Term','Score']]
    
    interm = interm.sort_values('Score', ascending =False)
    interm = interm.drop_duplicates()
    
    return ', '.join(interm.iloc[:nb_terms,0].tolist())

def format_topic_trend_for_display(search_topics_df) :
    
    agg_by_month = search_topics_df.groupby(by = [search_topics_df.index.year,search_topics_df.index.month] ).sum()
    agg_by_month['x_pos'] = [i for i in range(agg_by_month.shape[0])]
    topic_cols = agg_by_month.columns[:-2]
    for y,m in agg_by_month.index :
        for topic in topic_cols:
            topic_nr = int(topic.split('#')[1])
            agg_by_month.loc[(y,m),topic + '_terms'] = add_example_search_terms(search_topics_df,m,y,topic_nr,5)

    return agg_by_month

    
