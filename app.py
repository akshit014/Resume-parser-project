from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, Resume, Job, MatchScore
from parser import ResumeParser
from ranker import ResumeRanker
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# FIX: Use absolute path for database
basedir = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(basedir, 'database', 'resumes.db')

# Create database folder if it doesn't exist
os.makedirs(os.path.join(basedir, 'database'), exist_ok=True)
os.makedirs(os.path.join(basedir, 'uploads'), exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

db.init_app(app)
parser = ResumeParser()
ranker = ResumeRanker()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
with app.app_context():
    db.create_all()
    print("✅ Database initialized successfully!")
    print(f"📁 Database location: {database_path}")

@app.route('/')
def index():
    jobs = Job.query.all()
    stats = {
        'total_resumes': Resume.query.count(),
        'total_jobs': Job.query.count(),
    }
    return render_template('index.html', jobs=jobs, stats=stats)

@app.route('/upload', methods=['POST'])
def upload_resumes():
    if 'resumes' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('resumes')
    
    if not files or files[0].filename == '':
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    uploaded_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            parsed_data = parser.parse(filepath)
            
            if parsed_data:
                resume = Resume(
                    name=parsed_data['name'],
                    email=parsed_data['email'],
                    phone=parsed_data['phone'],
                    skills=parsed_data['skills'],
                    education=parsed_data['education'],
                    experience=parsed_data['experience'],
                    raw_text=parsed_data['raw_text']
                )
                db.session.add(resume)
                uploaded_count += 1
            
            os.remove(filepath)
    
    db.session.commit()
    
    flash(f'Successfully uploaded and parsed {uploaded_count} resume(s)!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/add-job', methods=['POST'])
def add_job():
    title = request.form.get('title')
    description = request.form.get('description')
    
    if not title or not description:
        flash('Please provide both job title and description', 'error')
        return redirect(url_for('index'))
    
    job = Job(title=title, description=description)
    db.session.add(job)
    db.session.commit()
    
    flash(f'Job "{title}" added successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/rank/<int:job_id>')
def rank_resumes_view(job_id):
    job = Job.query.get_or_404(job_id)
    resumes = Resume.query.all()
    
    if not resumes:
        flash('No resumes found. Please upload resumes first.', 'error')
        return redirect(url_for('index'))
    
    resume_data = [{
        'id': r.id,
        'raw_text': r.raw_text
    } for r in resumes]
    
    rankings = ranker.rank_resumes(resume_data, job.description)
    
    MatchScore.query.filter_by(job_id=job_id).delete()
    
    for resume_id, score in rankings:
        match = MatchScore(
            resume_id=resume_id,
            job_id=job_id,
            score=float(score)
        )
        db.session.add(match)
    
    db.session.commit()
    
    flash(f'Successfully ranked {len(resumes)} resumes for "{job.title}"!', 'success')
    return redirect(url_for('results', job_id=job_id))

@app.route('/results/<int:job_id>')
def results(job_id):
    job = Job.query.get_or_404(job_id)
    matches = MatchScore.query.filter_by(job_id=job_id).order_by(MatchScore.score.desc()).all()
    
    results = []
    for match in matches:
        resume = Resume.query.get(match.resume_id)
        results.append({
            'resume': resume,
            'score': round(match.score * 100, 2),
            'match': match
        })
    
    return render_template('results.html', job=job, results=results)

@app.route('/dashboard')
def dashboard():
    resumes = Resume.query.order_by(Resume.uploaded_at.desc()).all()
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    
    stats = {
        'total_resumes': Resume.query.count(),
        'total_jobs': Job.query.count(),
    }
    
    return render_template('dashboard.html', resumes=resumes, jobs=jobs, stats=stats)

@app.route('/delete-resume/<int:resume_id>', methods=['POST'])
def delete_resume(resume_id):
    resume = Resume.query.get_or_404(resume_id)
    MatchScore.query.filter_by(resume_id=resume_id).delete()
    db.session.delete(resume)
    db.session.commit()
    
    flash('Resume deleted successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/delete-job/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    job = Job.query.get_or_404(job_id)
    MatchScore.query.filter_by(job_id=job_id).delete()
    db.session.delete(job)
    db.session.commit()
    
    flash('Job deleted successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/clear-all', methods=['POST'])
def clear_all():
    MatchScore.query.delete()
    Resume.query.delete()
    Job.query.delete()
    db.session.commit()
    
    flash('All data cleared successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🚀 Starting Resume Parser Application...")
    print(f"📂 Project directory: {basedir}")
    print(f"💾 Database: {database_path}")
    print(f"📤 Uploads folder: {app.config['UPLOAD_FOLDER']}")
    print("\n🌐 Open your browser to: http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)