from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize database
db = SQLAlchemy()

# Resume table - stores parsed resume information
class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    skills = db.Column(db.Text)
    education = db.Column(db.Text)
    experience = db.Column(db.Text)
    raw_text = db.Column(db.Text)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Resume {self.name}>'

# Job table - stores job descriptions
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Job {self.title}>'

# MatchScore table - stores similarity scores
class MatchScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resume.id'))
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'))
    score = db.Column(db.Float)
    
    # Relationships
    resume = db.relationship('Resume', backref='match_scores')
    job = db.relationship('Job', backref='match_scores')
    
    def __repr__(self):
        return f'<MatchScore Resume:{self.resume_id} Job:{self.job_id} Score:{self.score}>'