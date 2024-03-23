import pandas as pd
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import numpy as np
import nltk
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
      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class maj:
  def search_word(self,sentence, word):
      sentence_lower = sentence.lower()
      word_lower = word.lower()
      if word_lower in sentence_lower:
          return True
      else:
          return False

  def set_seed(self,seed: int):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
  # self.set_seed(42)

  def postprocesstext(self,content):
    final=""
    for sent in sent_tokenize(content):
      sent = sent.capitalize()
      final = final +" "+sent
    return final
  summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
  summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
  summary_model = summary_model.to(device)
  
  def summarizer(self,text,model,tokenizer):
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
    summary = self.postprocesstext(summary)
    summary= summary.strip()

    return summary


  # summarized_text = summarizer(text,summary_model,summary_tokenizer)

  """# **Answer Span Extraction (Keywords and Noun Phrases)**"""
  def get_nouns_multipartite(self,content):
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



  def get_keywords(self,originaltext,summarytext):
    keywords = self.get_nouns_multipartite(originaltext)
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


  # imp_keywords = get_keywords(text,summarized_text)
  # print (imp_keywords)

  """# **Question generation with T5**"""
  def m(self):
    question_model = T5ForConditionalGeneration.from_pretrained('Koundinya-Atchyutuni/t5-end2end-questions-generation')
    question_model = question_model.to(device)
    return question_model
  def l(self):
    question_tokenizer = T5Tokenizer.from_pretrained('Koundinya-Atchyutuni/t5-end2end-questions-generation')
    return question_tokenizer

  def get_question(self,context,answer,model,tokenizer):
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
  
import pickle
obj=maj()
pickle.dump(obj,open('test.pkl','wb'))