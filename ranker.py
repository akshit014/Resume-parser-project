from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResumeRanker:
    def __init__(self):
        """Initialize TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)  # Use both single words and pairs
        )
    
    def rank_resumes(self, resumes, job_description):
        """
        Rank resumes against a job description
        
        Args:
            resumes: List of dictionaries with 'id' and 'raw_text'
            job_description: String of job description
            
        Returns:
            List of tuples (resume_id, score) sorted by score descending
        """
        if not resumes or not job_description:
            return []
        
        # Prepare texts: JD first, then all resumes
        texts = [job_description]
        resume_texts = []
        resume_ids = []
        
        for resume in resumes:
            resume_texts.append(resume['raw_text'])
            resume_ids.append(resume['id'])
        
        texts.extend(resume_texts)
        
        try:
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity between JD (index 0) and all resumes
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            # Extract scores (flatten from 2D to 1D array)
            scores = similarities[0]
            
            # Create list of (resume_id, score) tuples
            ranked_resumes = list(zip(resume_ids, scores))
            
            # Sort by score in descending order
            ranked_resumes.sort(key=lambda x: x[1], reverse=True)
            
            return ranked_resumes
            
        except Exception as e:
            print(f"Error ranking resumes: {e}")
            return [(resume_ids[i], 0.0) for i in range(len(resume_ids))]
    
    def get_top_keywords(self, text, n=10):
        """
        Extract top N keywords from text using TF-IDF
        Useful for showing why a resume matched
        """
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get indices of top scores
            top_indices = np.argsort(tfidf_scores)[-n:][::-1]
            
            # Get corresponding keywords
            top_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return top_keywords
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []