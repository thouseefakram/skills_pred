from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
import tensorflow as tf
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras import backend as K

# Define custom metrics
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())

# Load model, tokenizer, and MultiLabelBinarizer
model = tf.keras.models.load_model(
    r"C:\TF_Models\skills\tf_skills_model_v5.keras", 
    custom_objects={'precision': precision, 'recall': recall, 'f1_score': f1_score}
)
tokenizer = joblib.load(r"C:\TF_Models\skills\tf_skills_tokenizer_v5.pkl")
mlb = joblib.load(r"C:\TF_Models\skills\tf_skills_mlb_v5.pkl")

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Initialize Flask application
app = Flask(__name__, static_folder='static')

MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 150

def clean_and_lemmatize(text):
    text = text.lower()
    text = ' '.join([word.strip() for word in text.split() if word.strip() not in stop_words])
    text = re.sub(r'\s+', ' ', text)
    return text

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict_news_feed():
    job_title = request.form.get('Job_Title', '')
    industry = request.form.get('Industry', '')
    role_category = request.form.get('Role_Category', '')
    role = request.form.get('Role', '')
    
    input_text = f"{job_title} {role_category} {industry} {role}"
    processed_text = clean_and_lemmatize(input_text)
    
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    
    pred = model.predict(padded)
    
    def getPredOutput(pred, thresh=0.2):
        pred_bin = np.where(pred >= thresh, 1, 0)
        mlb_out = mlb.inverse_transform(pred_bin)[0]
        if mlb_out:
            return ", ".join(mlb_out)
        else:
            thresh -= 0.1
            if thresh < 0.1:
                return ""
            else:
                return getPredOutput(pred, thresh)
    
    pred_str = getPredOutput(pred, thresh=0.6)
    return jsonify({'prediction': pred_str})

if __name__ == '__main__':
    app.run(debug=True)
