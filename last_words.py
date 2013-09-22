#!/usr/bin/env python

from bs4 import BeautifulSoup
from urllib2 import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from collections import Counter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn

import operator

# imports are here because they crash together with matplotlib :-/
from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

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

def make_tag_cloud(data, can_be_noun_arg):
  words = ' '.join(list(data['last']))
  stop_words = sw.words()
  freqs = [(word, freq) for (word, freq) in get_tag_counts(words)
    if word not in stop_words and len(word)>2 and (can_be_noun_arg == can_be_noun(word))]
  freqs = freqs[:30]
  tags = make_tags(freqs, maxsize=80)
  fname = 'noun_last_words.png'
  if not can_be_noun_arg:
      fname = 'not_'+fname
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
  make_tag_cloud(data, False)
  #race_barplot(data)
  #age_hist(data)
  #race_pieplot(data)

if __name__=='__main__':
  main()
