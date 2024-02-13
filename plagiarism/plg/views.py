from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

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

    def jaccard_similarity(self, doc1, doc2):
        # Convert preprocessed text into sets of words
        words_doc1 = set(doc1.split())
        words_doc2 = set(doc2.split())
        
        # Calculate Jaccard similarity coefficient
        intersection = len(words_doc1.intersection(words_doc2))
        union = len(words_doc1.union(words_doc2))
        
        # Avoid division by zero
        if union == 0:
            return 0.0
        
        return intersection / union

    def get(self, request):
        # Return an empty response for GET requests
        return Response(status=status.HTTP_204_NO_CONTENT)

    def post(self, request):
        # Get text data from request
        text = request.data.get('text', '')

        # Perform text preprocessing
        preprocessed_text = self.preprocess_text(text)

        # Compare preprocessed text with reference text (you need to define reference text)
        reference_text = "This is the reference text for comparison."
        similarity_score = self.jaccard_similarity(preprocessed_text, reference_text)

        # Return similarity score as JSON response
        return Response({'similarity_score': similarity_score})
