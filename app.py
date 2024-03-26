from flask import Flask, render_template, request
import pandas as pd
import pickle
import en_SkillExtraction
import pymongo
import pandas as pd
import maj
import pke
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nlp = en_SkillExtraction.load()
import traceback
from flashtext import KeywordProcessor
from keybert import KeyBERT
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
client = pymongo.MongoClient("mongodb+srv://mulesaikrishnareddy2003:saikris2003@cluster0.bnz2azr.mongodb.net/subject_DB?retryWrites=true&w=majority")
def extract_keywords(text):
    # Initialize KeyBERT with a model
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    # Extract keywords from the text
    keywords = model.extract_keywords(text)
    # Return top 5 keywords
    return [keyword[0] for keyword in keywords][:5]
# import gensim
# from gensim.summarization import keywords

# Select the database
db = client["subject_DB"]
# Select the collection
collection = db["subjects"]
# Define global variables and load data
x = pickle.load(open('test.pkl', 'rb'))
# data = pd.read_excel("Merged_file.xlsx")
que = x.m()
qt = x.l()
x.set_seed(42)
# from rake_nltk import Rake
# def extract_keywords(text):
#     # Initialize RAKE
#     r = Rake()
#     # Extract keywords from the text
#     r.extract_keywords_from_text(text)
#     # Get the ranked keywords
#     ranked_keywords = r.get_ranked_phrases()
#     # Return top 5 keywords
#     return ranked_keywords[:5]
# Define routes
# def get_nouns_multipartite(text):
#     # content=str(content)
#     out=[]
#     extractor = pke.unsupervised.MultipartiteRank()
#     extractor.load_document(input=text,language='en')
#     print(text)
#     pos = {'PROPN','NOUN'}
#         #pos = {'PROPN','NOUN'}
#     stoplist = list(string.punctuation)
#     # stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
#     stoplist += stopwords.words('english')
#         # extractor.candidate_selection(pos=pos, stoplist=stoplist)
#     extractor.candidate_selection(pos=pos)
#     extractor.candidate_weighting(alpha=1.1,threshold=0.75,method='average')
#     keyphrases = extractor.get_n_best(n=15)
#     print(keyphrases)
#     for val in keyphrases:
#         out.append(val[0])
#     return out


# def get_keyword(text):
#     return keywords(text, words=5, split=True, scores=False)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')

def about():
      return render_template('about.html')

@app.route('/contact')

def contact():
      return render_template('contact.html')

@app.route('/main')

def main():
      return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    # lis = data['Topic']
    # w = list(request.form.values())[0]
    w=request.form['jobDescription']
    # indi = []
    st = str(w)
    doc = nlp(st)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    
    skills = list(set(skills))
    skills=[skill.lower() for skill in skills]
    # skills.append("java")
    # print(skills)
    # text="What is Java technology and why do I need it Java is a programming language and computing platform first released by Sun Microsystems in 1995. It has evolved from humble beginnings to power a large share of todays digital world, by providing the reliable platform upon which many services and applications are built. New, innovative products and digital services designed for the future continue to rely on Java, as well.While most modern Java applications combine the Java runtime and application together, there are still many applications and even some websites that will not function unless you have a desktop Java installed. Java.com, this website, is intended for consumers who may still require Java for their desktop applications – specifically applications targeting Java 8. Developers as well as users that would like to learn Java programming should visit the dev.java website instead and business users should visit oracle.com/java for more information."

    matching_documents = collection.find({"topic": {"$in": skills}})
    matching_documents_content=[]
    for document in matching_documents:
    # Print the keys present in the content array
        for content_item in document["content"]:
            matching_documents_content.append(content_item['value'])
   
    questions = []
    # # print(matching_documents_content)
    i=0
    for text in matching_documents_content:
        i+=1
        qs = []
        # print(text)
        # text = "Object-Oriented Programming is a paradigm that provides many concepts, such as inheritance, data binding, polymorphism, etc.Simula is considered the first object-oriented programming language. The programming paradigm where everything is represented as an object is known as a truly object-oriented programming language."
        summarized_text = x.summarizer(text, x.summary_model, x.summary_tokenizer)
                # print(summarized_text)
        keywords = extract_keywords(text)
        print ("keywords unsummarized: ",keywords)
        keyword_processor = KeywordProcessor()
        for keyword in keywords:
            keyword_processor.add_keyword(keyword)
            keywords_found = keyword_processor.extract_keywords(summarized_text)
            keywords_found = list(set(keywords_found))
        print ("keywords_found in summarized: ",keywords_found)

        important_keywords =[]
        for keyword in keywords:
            if keyword in keywords_found:
                important_keywords.append(keyword)
                # print(imp_keywords)
        for answer in important_keywords:
            ques = x.get_question(summarized_text, answer, que, qt) 
            qs.append(ques)
        qsz = set(qs)
        for z in qsz:
            questions.append(z)
        if(i==5):
            break    
    return render_template("main.html", prediction_text=questions)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
