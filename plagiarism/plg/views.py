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

import fitz
from rest_framework.response import Response
from rest_framework.views import APIView

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from nltk.tokenize import sent_tokenize
import numpy as np

from apify_client import ApifyClient


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
    

    # the text preprocessing and vectorzation are done to find the words with more weight, because this words are passing to APIFY to find the website realted to the input text
    
    #///////////////////////////////////////WEB SCRAPPING////////////////////////////////////////////////////////////////////////////////

    
    
    def get_urls_related_to_keywords(self,keywords):
        # Configure the Selenium Chrome WebDriver
        driver = webdriver.Chrome()

        # Open Google search
        driver.get("https://www.google.com")

        # Find the search input element and enter the keywords
        search_box = driver.find_element(By.NAME, "q")

        search_box.send_keys(keywords)
        search_box.send_keys(Keys.RETURN)

        # Wait for the search results to load
        time.sleep(2)

        # Get the URLs of the first 15 search results
        search_results = driver.find_elements(By.CSS_SELECTOR, ".tF2Cxc")

        urls = []
        for result in search_results[:2]:
            url_element = result.find_element(By.CSS_SELECTOR, "a")
            url = url_element.get_attribute("href")
            urls.append(url)

        # Close the browser
        driver.quit()

        return urls
        
        
    def scrape_urls(self,urls):
        scraped_data = []
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                soup = BeautifulSoup(response.content, "html.parser")
                # Get the text content within the <body> tag
                paragraphs = soup.find_all("p")
                body_text = "\n".join([p.get_text(separator='\n') for p in paragraphs])
                scraped_data.append({"url": url, "body_text": body_text})
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        return scraped_data

    
   

  


    #//////////////////////////// AI Algorithm//////////////////////////////
        


    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
    
    def preprocess_text_in_chunks(self, text, max_chunk_length=510):
        sentences = sent_tokenize(text)
        all_embeddings = []

        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            
            if not tokens:
                # Skip empty tokens
                continue
            
            chunk_size = min(max_chunk_length, len(tokens))
            chunk_tokens = tokens[:chunk_size]
            chunk = ["[CLS]"] + chunk_tokens + ["[SEP]"]

            input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(chunk)])

            with torch.no_grad():
                outputs = self.model(input_ids)
                embeddings = outputs[0][:, 1:-1, :].mean(dim=1)

            # Remove extra dimensions
            embeddings = embeddings.squeeze()

            all_embeddings.append(embeddings)

        if not all_embeddings:
            # Return zero embeddings if no embeddings were produced
            return torch.zeros(self.model.config.hidden_size).numpy()

        aggregated_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)

        return aggregated_embeddings.numpy()

    
    

    def check_plagiarism(self, input_text, scraped_data):
        similarities = []

        for scraped_item in scraped_data:
            scraped_text = scraped_item["body_text"]
            input_sentences = sent_tokenize(input_text)
            scraped_sentences = sent_tokenize(scraped_text)
            sentence_similarities = []

            for input_sentence in input_sentences:
                input_embeddings = self.preprocess_text_in_chunks(input_sentence)
                max_similarity = 0

                for scraped_sentence in scraped_sentences:
                    scraped_embeddings = self.preprocess_text_in_chunks(scraped_sentence)
                    similarity_matrix = cosine_similarity([input_embeddings], [scraped_embeddings])
                    similarity = similarity_matrix[0][0]
                    max_similarity = max(max_similarity, similarity)

                sentence_similarities.append(max_similarity)

            similarity = sum(sentence_similarities) / len(sentence_similarities) if len(sentence_similarities) > 0 else 0
            similarities.append({"url": scraped_item["url"], "similarity": similarity})

        return similarities
    
    



#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    def get(self, request):
        # Return an empty response for GET requests
        return Response(status=status.HTTP_204_NO_CONTENT)

    def post(self, request):
         # Get the PDF file from the request
        pdf_file = request.FILES.get('pdf_file')
        if pdf_file:
            if not pdf_file:
                return Response({'error': 'No PDF file uploaded'}, status=400)

            # Open the PDF file
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

            # Read the text from each page
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text()

            # Close the PDF file
            pdf_document.close()

            # Store the extracted text in a variable
            extracted_text = text

            print(extracted_text)

        else:
            extracted_text = request.data.get('text', '')

        
        # Perform text preprocessing
        preprocessed_text = self.preprocess_text(extracted_text)

       

        # Vectorize text
        tfidf_matrix, feature_names = self.vectorize_text([preprocessed_text])

        # Get top words
        top_words = self.get_top_words(tfidf_matrix, feature_names, n=10)

        query = ' '.join(top_words)
        

        
        keywords = query
        urls = self.get_urls_related_to_keywords(keywords)
        if urls is not None:
            for url in urls:
                print(url)
        else:
            urls = 'https://roadsafetycanada.com/'


        scraped_data = self.scrape_urls(urls)
        for data in scraped_data:
            print(data)
        # Print top words
        #print("Top words:", top_words)

        # Return similarity score as JSON response

        
        
        
        scraped_text= scraped_data

        similarity = self.check_plagiarism(extracted_text, scraped_data)
        
        print("Cosine Similarity (Deep Plagiarism):", similarity)
        
        similarity_threshold = .85
        # List to store URLs with similarity score above the threshold
        high_similarity_urls = []

        # Iterate over each similarity item
        for similarity_item in similarity:
            url = similarity_item["url"]
            score = similarity_item["similarity"]
            percentage = score * 100  # Convert similarity score to percentage
            if score > similarity_threshold:
                high_similarity_urls.append({"url": url, "similarity": percentage})


        return Response({'top_words': query, 'scrapped_data': scraped_text, 'similarity': high_similarity_urls})



