# -*- coding: utf-8 -*-
"""Copy of scoraproject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18XFGqCFyf9uRuvhpgYEVYgKpiRjuoV_T

## Installation of libraries
"""

# pip install --quiet flashtext==2.7
 #pip install git+https://github.com/boudinfl/pke.git

# pip install --quiet transformers==4.8.1
# !pip install --quiet sentencepiece==0.1.95
# !pip install --quiet textwrap3==0.9.2
# !pip install --quiet gradio==3.0.20

# !pip install --quiet strsim==0.0.3
# !pip install --quiet sense2vec==2.0.0

# # Commented out IPython magic to ensure Python compatibility.
# !pip install --quiet ipython-autotime
# # %load_ext autotime

# !pip install --quiet sentence-transformers==2.2.2

import pandas as pd
data=pd.read_csv("OS_dataset - Dataset.csv")

def search_word(sentence, word):
    sentence_lower = sentence.lower()
    word_lower = word.lower()

    if word_lower in sentence_lower:
        return True
    else:
        return False

lis=data['Topic']
w="os"
indi=[]
i=0
for t in lis:
  if search_word(t,w):
    indi.append(i)
  i=i+1
# indi

from textwrap3 import wrap

text = data['Text'][3]

for wrp in wrap(text, 150):
  print (wrp)
print ("\n")

"""Musk tweeted that his electric vehicle-making company tesla will not accept payments in bitcoin because of environmental concerns. He also said that
the company was working with developers of dogecoin to improve system transaction efficiency. The world's largest cryptocurrency hit a two-month low,
while doge coin rallied by about 20 percent. Musk has in recent months often tweeted in support of crypto, but rarely for bitcoin.

# **Summarization with T5**
"""

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final


def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary


summarized_text = summarizer(text,summary_model,summary_tokenizer)


print ("\noriginal Text >>")
for wrp in wrap(text, 150):
  print (wrp)
print ("\n")
print ("Summarized Text >>")
for wrp in wrap(summarized_text, 150):
  print (wrp)
print ("\n")

"""# **Answer Span Extraction (Keywords and Noun Phrases)**"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback

def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN','NOUN'}
        #pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)


        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out

from flashtext import KeywordProcessor


def get_keywords(originaltext,summarytext):
  keywords = get_nouns_multipartite(originaltext)
  print ("keywords unsummarized: ",keywords)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))
  print ("keywords_found in summarized: ",keywords_found)

  important_keywords =[]
  for keyword in keywords:
    if keyword in keywords_found:
      important_keywords.append(keyword)

  return important_keywords[:4]


imp_keywords = get_keywords(text,summarized_text)
print (imp_keywords)

"""# **Question generation with T5**"""

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question



for wrp in wrap(summarized_text, 150):
  print (wrp)
print ("\n")

for answer in imp_keywords:
  ques = get_question(summarized_text,answer,question_model,question_tokenizer)
  print (ques)
  print (answer.capitalize())
  print ("\n")

questins=[]
answers=[]
for i in indi:
  text = data['Text'][i]
  summarized_text = summarizer(text,summary_model,summary_tokenizer)
  imp_keywords = get_keywords(text,summarized_text)
  for answer in imp_keywords:
    ques = get_question(summarized_text,answer,question_model,question_tokenizer)
    # print (ques)
    questins.append(ques)
    answers.append(answer.capitalize())
    # print ("\n")

"""# **Filter keywords with Maximum marginal Relevance**"""

# !wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
# !tar -xvf  s2v_reddit_2015_md.tar.gz

import numpy as np
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('s2v_old')

from sentence_transformers import SentenceTransformer
# paraphrase-distilroberta-base-v1
sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')

from similarity.normalized_levenshtein import NormalizedLevenshtein
normalized_levenshtein = NormalizedLevenshtein()

def filter_same_sense_words(original,wordlist):
  filtered_words=[]
  base_sense =original.split('|')[1]
  print (base_sense)
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  score=[]
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)

def sense2vec_get_words(word,s2v,topn,question):
    output = []
    print ("word ",word)
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      # print (most_similar)
      output = filter_same_sense_words(sense,most_similar)
      print ("Similar ",output)
    except:
      output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)

    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

def get_distractors_wordnet(word):
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]

      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0:
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  distractors = sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
  print ("distractors ",distractors)
  if len(distractors) ==0:
    return distractors
  distractors_new = [word.capitalize()]
  distractors_new.extend(distractors)
  # print ("distractors_new .. ",distractors_new)

  embedding_sentence = origsentence+ " "+word.capitalize()
  # embedding_sentence = word
  keyword_embedding = sentencemodel.encode([embedding_sentence])
  distractor_embeddings = sentencemodel.encode(distractors_new)

  # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
  max_keywords = min(len(distractors_new),5)
  filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  # filtered_keywords = filtered_keywords[1:]
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() !=word.lower():
      final.append(wrd.capitalize())
  final = final[1:]
  return final

sent = "What cryptocurrency did Musk rarely tweet about?"
keyword = "Bitcoin"

# sent = "What did Musk say he was working with to improve system transaction efficiency?"
# keyword= "Dogecoin"


# sent = "What company did Musk say would not accept bitcoin payments?"
# keyword= "Tesla"


# sent = "What has Musk often tweeted in support of?"
# keyword = "Cryptocurrency"

print (get_distractors(keyword,sent,s2v,sentence_transformer_model,40,0.2))

x=input()
lis=data['Topic']
w=x
indi=[]
i=0
for t in lis:
  if search_word(t,w):
    indi.append(i)
  i=i+1

questins=[]
answers=[]
for i in indi:
  text = data['Text'][i]
  summarized_text = summarizer(text,summary_model,summary_tokenizer)
  imp_keywords = get_keywords(text,summarized_text)
  for answer in imp_keywords:
    ques = get_question(summarized_text,answer,question_model,question_tokenizer)
    # print (ques)
    questins.append(ques)
    answers.append(answer.capitalize())

import pickle

answers

questins