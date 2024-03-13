import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)


# Create Streamlit web app
st.title('Fake News Detector')
st.sidebar.header('User Input')
input_text = st.sidebar.text_area('Enter news article', '')
threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5, 0.05)

def predict(input_text):
    if input_text:
        input_data = vector.transform([input_text])
        prob_fake = model.predict_proba(input_data)[0, 1]
        return prob_fake

if input_text:
    prob_fake = predict(input_text)
    st.write('Probability of being Fake:', prob_fake)

    if prob_fake > threshold:
        st.error('This news is likely Fake.')
    else:
        st.success('This news is likely Real.')

# Add some explanations and tips
st.sidebar.markdown('**Tips:**')
st.sidebar.markdown('1. Enter a news article in the text area on the left.')
st.sidebar.markdown('2. Adjust the threshold to control sensitivity.')
st.sidebar.markdown('3. If the probability is above the threshold, it\'s likely Fake; otherwise, it\'s likely Real.')




