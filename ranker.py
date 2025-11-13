from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

class ResumeRanker:
    """
    Rank resumes against a job description using:
      - TF-IDF + cosine similarity (primary signal)
      - Skill overlap boost (secondary signal) using parsed skills from resumes
    
    Final score = alpha * tfidf_score + (1 - alpha) * skill_match
    where skill_match is in [0,1].
    
    The ranker returns a list of tuples:
      (resume_id, final_score, {'tfidf': tfidf_score, 'skill_match': skill_match})
    """
    def __init__(self, max_features=2000, ngram_range=(1,2), alpha=0.7):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range
        )
        # weight for TF-IDF score vs skill overlap
        self.alpha = float(alpha)

    def _safe_text(self, t):
        return t if (t is not None and isinstance(t, str)) else ""

    def _extract_top_terms_from_vector(self, tfidf_matrix_row, feature_names, top_n=30):
        """
        Given a TF-IDF row vector (1D array) and feature names, return top_n terms
        with positive tf-idf weight sorted descending.
        """
        arr = np.array(tfidf_matrix_row).ravel()
        if arr.size == 0:
            return []
        # get indices of top values
        top_idx = np.argsort(arr)[-top_n:][::-1]
        terms = []
        for i in top_idx:
            if arr[i] > 0:
                terms.append(feature_names[i])
        return terms

    def _compute_skill_match(self, job_terms, resume_skills_list):
        """
        Compute a simple skill overlap ratio in [0,1].
        job_terms: list of words/phrases representing JD important terms
        resume_skills_list: list of skills parsed from resume (normalized)
        """
        if not job_terms or not resume_skills_list:
            return 0.0

        # Normalize lowercase and simple stripping
        job_set = set([t.lower().strip() for t in job_terms if t and isinstance(t, str)])
        resume_set = set([s.lower().strip() for s in resume_skills_list if s and isinstance(s, str)])

        if not job_set:
            return 0.0

        common = job_set.intersection(resume_set)
        # Ratio: common / job requirements (if job has many terms, one match counts less)
        match_ratio = len(common) / len(job_set)
        # ensure within 0..1
        return min(1.0, max(0.0, float(match_ratio)))

    def rank_resumes(self, resumes, job_description, top_job_terms=30):
        """
        Args:
            resumes: list of dicts; each dict should have:
                - 'id' : int
                - 'raw_text' : str
                - optionally 'skills_list' : list of skill strings (preferred)
            job_description: string

        Returns:
            list of tuples (resume_id, final_score, meta) sorted by final_score desc
            Where meta is dict: {'tfidf': <0..1>, 'skill_match': <0..1>}
        """
        # Basic guards
        if not resumes or not job_description:
            return []

        # Build texts: JD first then resumes
        jd_text = self._safe_text(job_description)
        resume_texts = []
        resume_ids = []
        resume_skills_lists = []

        for r in resumes:
            resume_ids.append(r.get('id'))
            resume_texts.append(self._safe_text(r.get('raw_text', '')))
            # Accept skills_list if provided, else empty list
            skl = r.get('skills_list') or []
            # If skills_list is a comma string, split it
            if isinstance(skl, str):
                skl = [s.strip() for s in skl.split(',') if s.strip()]
            resume_skills_lists.append([s for s in skl if s])

        texts = [jd_text] + resume_texts

        try:
            # Fit TF-IDF on current JD + resumes (vocab adaptive to the set)
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()

            # JD vector is at index 0, resumes start at index 1
            jd_vector = tfidf_matrix[0:1]
            resumes_matrix = tfidf_matrix[1:]

            # Cosine similarities: shape (1, n_resumes)
            similarities = cosine_similarity(jd_vector, resumes_matrix)  # 2D
            tfidf_scores = similarities.ravel()  # 1D array length == n_resumes

            # Derive prominent job terms from JD TF-IDF (for skill overlap)
            jd_tfidf_arr = jd_vector.toarray()[0]
            job_terms = self._extract_top_terms_from_vector(jd_tfidf_arr, feature_names, top_n=top_job_terms)

            ranked = []
            for i, rid in enumerate(resume_ids):
                tfidf_score = float(tfidf_scores[i]) if i < len(tfidf_scores) else 0.0
                # compute skill match
                skill_match = self._compute_skill_match(job_terms, resume_skills_lists[i])
                # combine
                final_score = (self.alpha * tfidf_score) + ((1.0 - self.alpha) * skill_match)
                # ensure numeric safety
                if math.isnan(final_score) or final_score < 0:
                    final_score = 0.0
                ranked.append((rid, float(final_score), {'tfidf': float(tfidf_score), 'skill_match': float(skill_match)}))

            # sort by final_score desc
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked

        except Exception as e:
            # On exception, log and return zeros with fallback meta
            print(f"[ResumeRanker] Error ranking resumes: {e}")
            fallback = []
            for i, rid in enumerate(resume_ids):
                fallback.append((rid, 0.0, {'tfidf': 0.0, 'skill_match': 0.0}))
            return fallback

    def get_top_keywords(self, text, n=10):
        """
        Return top n keywords for a text by fitting TF-IDF on the single text.
        (Useful for quick debugging, but for production you generally want to fit
        on a larger corpus.)
        """
        try:
            txt = self._safe_text(text)
            if not txt:
                return []
            v = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
            tfidf = v.fit_transform([txt])
            feature_names = v.get_feature_names_out()
            arr = tfidf.toarray()[0]
            top_idx = np.argsort(arr)[-n:][::-1]
            keywords = [feature_names[i] for i in top_idx if arr[i] > 0]
            return keywords
        except Exception as e:
            print(f"[ResumeRanker] Error extracting keywords: {e}")
            return []
    