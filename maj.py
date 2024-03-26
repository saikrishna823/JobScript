import pandas as pd
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import numpy as np
import nltk
# import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
# import nltk
nltk.download('averaged_perceptron_tagger')
import string
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
# data=pd.read_csv("OS_dataset - Dataset.csv")
# import spacy
# nlp=spacy.load("en_core_web_sm")      
# import en_core_web_sm
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class maj:
  # def search_word(self,sentence, word):
  #     sentence_lower = sentence.lower()
  #     word_lower = word.lower()
  #     if word_lower in sentence_lower:
  #         return True
  #     else:
  #         return False

  def set_seed(self,seed: int):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

  def postprocesstext(self,content):
    final=""
    for sent in sent_tokenize(content):
      sent = sent.capitalize()
      final = final +" "+sent
    return final
  
  summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
  summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
  # summary_model = summary_model.to(device)
  
  def summarizer(self,text,model,tokenizer):
    text = text.strip().replace("\n"," ")
    text = "summarize: "+text
    # print (text)
    max_len = 512
    encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

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
    summary = self.postprocesstext(summary)
    summary= summary.strip()
    return summary
  
  
  # def get_nouns_from_text(self,text):
  #   # Tokenize the text into words
  #   words = word_tokenize(text)
    
  #   # Tag the words with their part-of-speech (POS)
  #   tagged_words = pos_tag(words)
    
  #   # Define a list of allowed POS tags for nouns
  #   allowed_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    
  #   # Filter out nouns based on their POS tags
  #   nouns = [word for word, tag in tagged_words if tag in allowed_tags]
    
  #   # Remove stopwords and punctuation
  #   stop_words = set(stopwords.words('english'))
  #   punctuation = set(string.punctuation)
    
  #   cleaned_nouns = [word for word in nouns if word.lower() not in stop_words and word not in punctuation]
    
  #   return cleaned_nouns

  # imp_keywords = get_keywords(text,summarized_text)
  # print (imp_keywords)

  def m(self):
    question_model = T5ForConditionalGeneration.from_pretrained('Koundinya-Atchyutuni/t5-end2end-questions-generation')
    # question_model = question_model.to(device)
    return question_model
  
  def l(self):
    question_tokenizer = T5Tokenizer.from_pretrained('Koundinya-Atchyutuni/t5-end2end-questions-generation')
    return question_tokenizer

  def get_question(self,context,answer,model,tokenizer):
    text = "context: {} answer: {}".format(context,answer)
    encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt")
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
  
import pickle
obj=maj()
pickle.dump(obj,open('test.pkl','wb'))