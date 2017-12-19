
# coding: utf-8

# # Build corpus

import numpy as np
import pandas as pd
import json
import datetime
import requests
from bs4 import BeautifulSoup
import os
import re
import codecs
from tqdm import tqdm


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

def googleSearch(session, txt_query):
    
    # The headers parameters have been adapted to the header I send when querying Google
    headers_Get = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'fr,ja-JP;q=0.9,ja;q=0.8,en-US;q=0.7,en;q=0.6,nl;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
        }

    q = '+'.join(txt_query.split())
    url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
    r = session.get(url, headers=headers_Get)

    soup = BeautifulSoup(r.text, "html.parser")
    output = []
    for searchWrapper in soup.find_all('span', {'class':'st'}): #this line may change in future based on google's web page structure
        desc = searchWrapper.text
        output.append(desc)

    return output

def build_corpus(batch_size, queries, out_folder) :
    try :
        os.mkdir(out_folder)
    except :
        pass

    loop_range = range((len(queries))/batch_size+1)

    for i in tqdm(loop_range) :
    
        if  (i < 4) or (i > 60) :
            # Make search and save response by batches
            if i != loop_range[-1] :
                bottom = i*batch_size
                top = i*batch_size+batch_size
            else :
                bottom = i*batch_size
                top = len(queries)

            queries_subset = queries[bottom:top]
        
            # Getting rid of the google maps searches
            queries_subset = [q for q in queries_subset if not (re.search('->',q) or re.search(',',q))]
        
            # Open session and build corpus
            s = requests.Session()
            documents =[{q : googleSearch(s,q)} for q in queries_subset]
        
            # Save data to a file
            filename = out_folder + "batch_" + str(i) + ".txt"
            myfile = codecs.open(filename, "w", "utf-8")


            out = json.dumps(documents, ensure_ascii=False)
            myfile.write("%s\n" % out )
        
            myfile.close()


# Load query data
indir = 'Recherches/'
search_data = load_search_jsons(indir)


# Build corpus
# Only build corpus for queries done more than once (for now)

# sel = search_data['query_text'].value_counts()>1
# non_unique_queries = search_data['query_text'].value_counts()[sel].index.tolist()
# print len(non_unique_queries)

# Build the rest of the corpus
sel = search_data['query_text'].value_counts()==1
unique_queries = search_data['query_text'].value_counts()[sel].index.tolist()
print len(unique_queries)

# See line 82!! condition to delete
build_corpus(10, unique_queries,"Recherches/NLP_moredata/")



