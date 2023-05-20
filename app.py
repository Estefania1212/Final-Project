import pandas as pd
import nltk
import re

from tensorflow import keras
import streamlit as st
from PIL import Image
from keras.models import Sequential 
from keras.utils import pad_sequences
import pickle

from nltk.corpus import stopwords
from nltk import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

namefile = 'logistic_regression_m.pkl'
model = pickle.load(open(namefile, 'rb'))

namefile2 = 'logistic_regression_vec.pkl'
vectorizer = pickle.load(open(namefile2, 'rb'))


def clean_reviews(review):
    
    rev = str(review)
    cleaned_review  = rev
    
    # if url links then skip
    if re.match("(\w+:\/\/\S+)", rev) == None:
        
        # remove hashtag, @mention, emoji and image URLs
        rev = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ",rev).split())

        # remove punctuation
        rev = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", rev).split())

        # stop words and tokenization
        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(rev)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        rev = ' '.join(filtered_sentence)

        cleaned_review=rev

    return cleaned_review

##set page configuration
###python -m streamlit run TRY.py
st.set_page_config(page_title='Fake product detector',layout='wide')

##add page title and content
st.title('Fake Product Detector using Logistic Regression')

##add image
##image=Image.open(r'C:\Users\brill\OneDrive\Documents\DScourse\Fake reviews\fakereviewsfoto.jpg')
##st.image(image,use_column_width=True)

##get user input
review_text=st.text_input('Please enter your review to analyse:')

if st.button('predict'):
    review_text_cleaned=clean_reviews(review_text)
    myList=[review_text_cleaned]
    x_input=pd.DataFrame(myList,columns=['reviewTokenized'])
    X_input_tfidf = vectorizer.transform(x_input['reviewTokenized'])
   
    prediction=model.predict(X_input_tfidf)[0]

    if prediction==0:
        st.write('This is a fake review')
    else:
        st.write('This is a real review')
