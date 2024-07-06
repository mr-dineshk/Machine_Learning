from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    
    statement = request.form['inputtext']
    statement = statement.lower()  # Convert input text to lowercase
    vectorizer = TfidfVectorizer(max_features=5000)
    input_text_vectorized = vectorizer.transform([statement])  # Vectorize input text
    prediction = model.predict(input_text_vectorized)[0]  # Make prediction
    return render_template("index.html", statement=prediction) 

if __name__ == '__main__':
    app.run(debug= True)
