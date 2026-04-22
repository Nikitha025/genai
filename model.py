import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def is_followup(user_input, history):
    """
    Detects if the user input aims to continue the conversation or start a new career search.
    """
    if not history:
        return False
        
    lower_input = user_input.lower()
    restart_keywords = ["restart", "start over", "new query", "reset", "different career"]
    
    for kw in restart_keywords:
        if kw in lower_input:
            return False
            
    # By default, if conversation history exists, we treat it as an ongoing follow-up.
    return True


class CareerRetriever:
    def __init__(self, data_path='data/careers.csv'):
        """
        Initializes the TF-IDF vectorizer and loads the dataset.
        """
        self.data_path = data_path
        self.df = pd.DataFrame()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """
        Loads the definitions and fits the TFIDF vectorizer.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        
        # Combine skills and description into a single 'text' column for better context matching
        self.df['combined_features'] = self.df['skills'].fillna('') + " " + self.df['description'].fillna('')
        
        # Fit vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])

    def retrieve_careers(self, user_query, top_n=3):
        """
        Retrieves the top N career roles closely matching the user query using cosine similarity.
        """
        if not user_query.strip():
            return []

        # Vectorize user query
        query_vec = self.vectorizer.transform([user_query])
        
        # Calculate Cosine Similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top indices
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Ignore if similarity is 0 (no match at all) - allow returning top3
        retrieved_careers = []
        for idx in top_indices:
            career_dict = {
                "Role": self.df.iloc[idx]['career'],
                "Skills Needed": self.df.iloc[idx]['skills'],
                "Description": self.df.iloc[idx]['description'],
                "Roadmap": self.df.iloc[idx].get('roadmap', ''),
                "Match Score": float(similarities[idx])
            }
            retrieved_careers.append(career_dict)
                
        return retrieved_careers
