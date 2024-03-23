from flask import Flask, render_template, request
import pandas as pd
import pickle
import en_SkillExtraction
import maj
nlp = en_SkillExtraction.load()
app = Flask(__name__)

# Define global variables and load data
x = pickle.load(open('test.pkl', 'rb'))
data = pd.read_excel("Merged_file.xlsx")
que = x.m()
qt = x.l()

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    lis = data['Topic']
    w = list(request.form.values())[0]
    indi = []
    st = str(w)
    doc = nlp(st)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    skills = list(set(skills))
    print(skills)
    
    for z in skills:
        i = 0
        for t in lis:
            if x.search_word(str(t), z):
                indi.append(i)
                break
            i = i + 1
    indi = indi[1:8]
    questions = []
    for i in indi:
        qs = []
        text = data['Text'][i]
        summarized_text = x.summarizer(text, x.summary_model, x.summary_tokenizer)
        imp_keywords = x.get_keywords(text, summarized_text)
        for answer in imp_keywords:
            ques = x.get_question(summarized_text, answer, que, qt)
            ques = ques + "__ANS:" + answer
            qs.append(ques)
        qsz = set(qs)
        for z in qsz:
            questions.append(z)
    return render_template("index.html", prediction_text=questions)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
