from flask import Flask, render_template, request, jsonify
import pickle
import spacy  # Import spaCy

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl', 'rb'))

# Load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)[0]  # Get the first prediction
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
