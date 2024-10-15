import streamlit as st
import PyPDF2
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# For text preprocessing (if necessary)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf


# To download the Punkt tokenizer model
nltk.download('punkt')
nltk.download('punkt_tab')
# To download the list of stopwords
nltk.download('stopwords')

# To download the WordNet lexical database
nltk.download('wordnet')



# Dummy function to detect if text is AI generated (Replace with actual model predictions)
def predict_text_type(text, model_type='Logistic Regression'):
    # This is where you'd load your pre-trained model and vectorizer
    if model_type == 'Naive Bayes':
        model = pickle.load(open('naive_bayes.pkl', 'rb'))  # Load your Naive Bayes model
    elif model_type == 'Logistic Regression':
        model = pickle.load(open('logistic_regression.pkl', 'rb'))  # Load your Logistic Regression model
    else:
        # model_file = pickle.load(open('RNN.pkl', 'rb'))  # Load your RNN model
        model = tf.keras.models.load_model('RNN.h5')
        # with open("RNN.pkl", "rb") as f:
        #     model = pickle.load(f)

    # Example preprocessing of text
    vectorizer = pickle.load(open('tf_idf.pkl', 'rb'))  # Load the vectorizer for the model
    processed_text = vectorizer.transform([text]).toarray()     

    if not model_type == 'RNN':
        # Get the prediction
        prediction = model.predict(processed_text)
        
        # Example confidence score (modify according to your model output)
        confidence = model.predict_proba(processed_text)[0]

        # Explainability 
        if model_type == 'Logistic Regression':
            feature_names = vectorizer.get_feature_names_out()
            coef = model.coef_[0]
            word_indices = processed_text.nonzero()[1]
            words_in_text = [feature_names[i] for i in word_indices]
            word_contributions = [coef[i] for i in word_indices]
            word_contributions = sorted(zip(words_in_text, word_contributions), key=lambda x: x[1], reverse=True)
            if len(word_contributions) > 10:
                word_contributions_trunc = word_contributions[:5] + word_contributions[-5:]
                words, contributions = zip(*word_contributions_trunc)
            else:
                words, contributions = zip(*word_contributions)
            plt.figure(figsize=(10, 6))
            colors = ['green' if c > 0 else 'red' for c in contributions]
            plt.barh(words, contributions, color=colors)
            plt.xlabel('Contribution to Classification')
            plt.title('Significant Contributors to Logistic Regression Classification')
            plt.gca().invert_yaxis()
            # st.pyplot(plt)
        elif model_type == 'Naive Bayes':
            feature_names = vectorizer.get_feature_names_out()
            log_probs = model.feature_log_prob_
            pos_class_log_probs = log_probs[1]
            neg_class_log_probs = log_probs[0]
            word_indices = processed_text.nonzero()[1]
            words_in_text = [feature_names[i] for i in word_indices]
            word_contributions_pos = [pos_class_log_probs[i] for i in word_indices]
            word_contributions_neg = [neg_class_log_probs[i] for i in word_indices]
            word_contributions = list(zip(words_in_text, word_contributions_pos, word_contributions_neg))
            word_contributions = sorted(word_contributions, key=lambda x: abs(x[1] - x[2]), reverse=True)
            word_contributions = word_contributions[:10]  # top 10
            words, _, _ = zip(*word_contributions)
            plt.figure(figsize=(10, 6))
            pos_contributions = [c[1] for c in word_contributions]  # Positive class log-probs
            neg_contributions = [c[2] for c in word_contributions]  # Negative class log-probs
            contributions = np.array(pos_contributions) - np.array(neg_contributions)
            colors = ['green' if c > 0 else 'red' for c in contributions]
            plt.barh(words, contributions, color=colors)
            plt.xlabel('Difference in Log-Probabilities (Positive Class - Negative Class)')
            plt.title('Most Decisive words to Naive Bayes Classification')
            plt.gca().invert_yaxis()

        return 'AI Generated' if prediction == 1 else 'Human Written', max(confidence) * 100, plt

    else:
        processed_text_reshaped = processed_text.reshape(processed_text.shape[0], 1, processed_text.shape[1])
        prediction = model.predict(processed_text_reshaped)

        return 'AI Generated' if prediction > 0.5 else 'Human Written', prediction.item()*100 if prediction > 0.5 else (1-prediction.item())*100, None

    
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def preprocess_text(text):

    # To convert the text to lowercase
    text = text.lower()

    # To remove HTML tags from the text
    text = re.sub('<.*?>', '', text)

    # To remove URLs from the text
    text = re.sub(r'http\S+', '', text)

    # To remove special characters and numbers from the text
    text = re.sub('[^a-zA-Z\s]', '', text)

    # To tokenize the text
    tokens = word_tokenize(text)

    # To remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # To lemmatize each token
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # To rejoin tokens into a single string
    processed_text = ' '.join(tokens)

    return processed_text


# Streamlit Interface
st.title("AI vs Human Text Detector")

# Option to upload PDF or input text
option = st.selectbox("Choose input method:", ("Write Text", "Upload PDF"))

# If PDF upload is selected
if option == "Upload PDF":
    text = ""
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        #st.write("Extracted Text from PDF:")
        with st.expander('Expand to see extracted Text from PDF'):
            st.write(text)

# If text input is selected
else:
    text = st.text_area("Enter your text here:")


# Once text is available, proceed with prediction
if not text == "":
    processed_text = preprocess_text(text)
    with st.expander("See processed text"):
        st.write(processed_text)
    # Choose the model for detection
    model_choice = st.selectbox("Choose a model for prediction:", ("Naive Bayes", "Logistic Regression", "RNN"))
    if st.button("Detect Text Type"):
        result, confidence, plt = predict_text_type(processed_text, model_choice)
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}%")
        if not plt is None:
            st.pyplot(plt)
else:
    pass

