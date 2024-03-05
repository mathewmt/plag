from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from apify_client import ApifyClient
import requests
from bs4 import BeautifulSoup

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PreprocessTextView(APIView):
    def preprocess_text(self, text):
        # Tokenize text
        tokens = word_tokenize(text)
        # Convert to lowercase
        tokens = [word.lower() for word in tokens]
        # Remove punctuation and non-alphanumeric characters
        tokens = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in tokens]
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    
    def vectorize_text(self, preprocessed_texts):
        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        # Fit-transform the preprocessed texts to get the TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
        # Return the TF-IDF matrix and the feature names (words)
        return tfidf_matrix, vectorizer.get_feature_names_out()

    def get_top_words(self, tfidf_matrix, feature_names, n=10):
        # Get the average TF-IDF weights for each word across all texts
        avg_tfidf_weights = np.mean(tfidf_matrix.toarray(), axis=0)
        # Get the indices of the top N words with highest average TF-IDF weights
        top_indices = np.argsort(avg_tfidf_weights)[::-1][:n]
        # Get the words corresponding to the top indices
        top_words = [feature_names[i] for i in top_indices]
        # Return the top words
        return top_words
    
    

    def scrape_website(self,url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Get the title of the page
        title = soup.title.string if soup.title else "Title not found"
        
        # Get the main content of the page
        main_content = soup.find("div", class_="elementor-column elementor-col-50 elementor-top-column elementor-element elementor-element-b22f800")
        content = main_content.get_text() if main_content else "Main content not found"
        
        return title, url, content

  


    
        


    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
    
    def preprocess_text(self, text):
        # Tokenize input text
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(input_ids)
            embeddings = outputs[0][:, 1:-1, :].mean(dim=1)  # Average pooling of token embeddings
        return embeddings.numpy()
    
    def check_plagiarism(self, text1, text2):
        embeddings1 = self.preprocess_text(text1)
        embeddings2 = self.preprocess_text(text2)
        
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        return similarity

# Example usage


    def get(self, request):
        # Return an empty response for GET requests
        return Response(status=status.HTTP_204_NO_CONTENT)

    def post(self, request):
        # Get text data from request
        text = request.data.get('text', '')

        # Perform text preprocessing
        preprocessed_text = self.preprocess_text(text)

       

        # Vectorize text
        #tfidf_matrix, feature_names = self.vectorize_text([preprocessed_text])

        # Get top words
        #top_words = self.get_top_words(tfidf_matrix, feature_names, n=10)

        #query = ' '.join(top_words)
        



        # Print top words
        #print("Top words:", top_words)

        # Return similarity score as JSON response

        url = "https://roadsafetycanada.com/"
        title, url, content = self.scrape_website(url)
        

        

        text1 = text
        text2 = content

        similarity = self.check_plagiarism(text1, text2)
        print("Cosine Similarity (Deep Plagiarism):", similarity)

        
        return Response({'top Words': content})

