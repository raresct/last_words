#!/usr/bin/env python

from bs4 import BeautifulSoup
from urllib2 import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.feature_extraction.text  import TfidfTransformer

from collections import Counter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn

import operator

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

from pprint import pprint

def write_data(fname):
    with open(fname, 'w') as f:
        f.write('Statement|Age|Date|Race\n')
        base_url = 'http://www.tdcj.state.tx.us/death_row/'
        html_doc = urlopen(base_url+'dr_executed_offenders.html')
        soup = BeautifulSoup(html_doc)
        for e in soup.find_all('br'):
            e.extract()
        main_table = soup.find('table')
        rows = main_table.find_all('tr')
        start = 0
        for idx,row in enumerate(rows[1+start:]):
            print idx
            cols = row.find_all('td')
            last_words_url = base_url+cols[2].find('a')['href']
            last_words_doc = urlopen(last_words_url)
            last_words_soup = BeautifulSoup(last_words_doc)
            for e in last_words_soup.find_all('br'):
                e.extract()
            main_div = last_words_soup.find('div', {'id':'body'})
            paragraphs = main_div.find_all('p')
            [before_last_p] = filter(lambda x: x.text.strip()=='Last Statement:', paragraphs)
            last_statement = ' '.join([x.text for x in before_last_p.find_all_next('p')]).replace('\r','').replace('\n','')
            age = cols[6].string.strip()
            date = cols[7].string.strip()
            race = cols[8].string.strip()
            csv_row = '|'.join([last_statement, age, date, race])+'\n'
            f.write(csv_row.encode('ascii', 'ignore'))

def can_be_noun(test_word):
    synsets = wn.synsets(test_word)
    if len(synsets) == 0:
        return True
    for s in synsets:
        if s.pos == 'n':
            return True
    return False

def read_data(fname):
    data = pd.read_csv(fname, sep='|', header=0,
            names = ['last', 'age', 'date', 'race'])
    l = list(data['last'])
    default_str = 'This offender declined to make a last statement.'
    data['last'] = data['last'].str.replace(default_str, '')
    return data

def concatenate(data, can_be_noun_arg, stop_words):
    words = ' '.join(list(data['last']))
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform([words])
    #print len(count_vectorizer.vocabulary_)
    #print counts.todense().shape[1]
    can_be_noun_filter = lambda x: can_be_noun(x) if can_be_noun_arg else lambda x: not can_be_noun(x)
    voc = {k:counts[0,v] for k,v in count_vectorizer.vocabulary_.items() if can_be_noun_filter(k) and k not in stop_words}
    freqs = list(voc.items())
    #print freqs
    return sorted(freqs, key = lambda x: x[1], reverse=True)
    #counts_red = counts[:,list(voc.itervalues())]
    #print len(voc)
    #print counts_red.todense().shape[1]

    # old definition
    #freqs = [(word, freq) for (word, freq) in get_tag_counts(words)
    #    if word not in stop_words and len(word)>2 and (can_be_noun_arg == can_be_noun(word))]
    #return freqs

def freq_weight(data, can_be_noun_arg, stop_words):
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(data['last'])
    overall_freqs = np.mean(counts.todense(), 0)
    can_be_noun_filter = lambda x: can_be_noun(x) if can_be_noun_arg else not can_be_noun(x)
    voc = {k:overall_freqs[0,v] for k,v in count_vectorizer.vocabulary_.items() if can_be_noun_filter(k) and k not in stop_words}
    freqs = list(voc.items())
    return sorted(freqs, key = lambda x: x[1], reverse=True)


# # old definition without sklearn
#def freq_weight(data, can_be_noun_arg, stop_words):
#    probs = {}
#    for document in data['last']:
#        freqs = [(word, freq) for (word, freq) in get_tag_counts(document)
#            if word not in stop_words and len(word)>2 and (can_be_noun_arg == can_be_noun(word))]
#        total = np.sum([freq for (_, freq) in freqs])
#        for (word, freq) in freqs:
#            if word in probs:
#                probs[word] += np.float(freq)/total
#            else:
#                probs[word] = np.float(freq)/total
#        #pprint(probs)
#        freqs_unnorm = list(probs.items())
#        # normalize
#        sd_probs = np.std([prob for (_, prob) in freqs_unnorm])
#        freqs  = [(word , prob/sd_probs) for (word,prob) in freqs_unnorm]
#        freqs = sorted(freqs, key = lambda x: x[1], reverse=True)
#    return freqs

def race_tfidf(data, can_be_noun_arg, stop_words):
    print 
    data = data.groupby('race')['last']
    data = dict(list(data))
    docs = []
    for k in data:
        docs.append(' '.join(data[k]))
    count_vectorizer = CountVectorizer(stop_words='english')
    counts = count_vectorizer.fit_transform(docs)
    #print counts.todense().shape
    tfidf = TfidfTransformer(norm="l2", sublinear_tf='True')
    tfidf.fit(counts)
    #print "IDF:", tfidf.idf_.shape
    tf_idf_matrix = tfidf.transform(counts)
    freqs = {}
    sorted_voc = sorted(count_vectorizer.vocabulary_.iteritems(), key=operator.itemgetter(1))
    terms,_ = zip(*sorted_voc)
    for i,k in enumerate(data.keys()):
        # make list
        row = np.array(tf_idf_matrix.todense()[i,:])[0].tolist()
        freq = zip(terms, row)
        freqs[k] = sorted(freq, reverse=True, key=lambda x: x[1])
        print freqs[k][:5]
    #print tf_idf_matrix.todense().shape
    return freqs
 
def make_tag_cloud(data, can_be_noun_arg, process_option='freqs'):
    stop_words = sw.words()
    process_f = {
            'concatenate': lambda : concatenate(data, can_be_noun_arg, stop_words),
            'freqs': lambda : freq_weight(data, can_be_noun_arg, stop_words),
            'race' : lambda : race_tfidf(data, can_be_noun_arg, stop_words)
    }
    freqs = process_f[process_option]()
    if type(freqs) == type([]):
        freqs = freqs[:30]
        # normalize freqs in case they are counts
        sum_freqs = np.sum(x for _,x in freqs)
        freqs = [(w, np.float(f)/sum_freqs) for w,f in freqs]
        #pprint(freqs)
        #return
        tags = make_tags(freqs, maxsize=80)
        fname = 'noun_last_words_{}.png'.format(process_option)
        if not can_be_noun_arg:
            fname = 'not_'+fname
        create_tag_image(tags, fname, size=(900, 600), fontname='Lobster')
    elif type(freqs)==type({}):
        for k in freqs:
            top_freqs = freqs[k][:30]
            # normalize    
            sum_freqs = np.sum(x for _,x in top_freqs)
            top_freqs = [(w, np.float(f)/sum_freqs) for w,f in top_freqs]
            print top_freqs
            tags = make_tags(top_freqs, maxsize=15)
            fname = 'noun_last_words_{}_{}.png'.format(process_option,k)
            create_tag_image(tags, fname, size=(900, 600), fontname='Lobster')

def race_barplot(data):
    data_race = data.dropna(subset=['race'], how='all')
    race = data_race['race']
    race = Counter(race)
    sorted_race = sorted(race.iteritems(), key=operator.itemgetter(1))
    [races, counts] = zip(*sorted_race)
    race_range = np.arange(len(counts))
    plt.yticks(race_range, races)
    plt.ylim((np.min(race_range)-1, np.max(race_range)+1))
    plt.barh(race_range, counts, align='center', height=0.1)
    plt.show()

def age_hist(data):
    _, bins, _ = plt.hist(data['age'], bins=15)
    bins = [round(x) for x in bins]
    plt.xticks(bins)
    plt.ylim((0,100))
    plt.show()

def race_pieplot(data):
    data_race = data.dropna(subset=['race'], how='all')
    race = data_race['race']
    race = Counter(race)
    races = race.keys()
    counts = race.values()
    total_counts = np.sum(counts)
    fracs = [x*100/total_counts for x in counts]
    plt.pie(fracs, labels=races)
    plt.show()

def main():
    fname = 'last_words_data.csv'
    #write_data(fname)
    data = read_data(fname)
    make_tag_cloud(data, can_be_noun_arg=True, process_option='race')
    #race_barplot(data)
    #age_hist(data)
    #race_pieplot(data)

if __name__=='__main__':
    main()
