from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, Resume, Job, MatchScore
from parser import ResumeParser
from ranker import ResumeRanker
import os
from werkzeug.utils import secure_filename
import json

# Production config
if os.environ.get('RENDER'):
    basedir = '/opt/render/project/src'
else:
    basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Paths
basedir = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(basedir, 'database', 'resumes.db')
os.makedirs(os.path.join(basedir, 'database'), exist_ok=True)
os.makedirs(os.path.join(basedir, 'uploads'), exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

db.init_app(app)
parser = ResumeParser()
# instantiate ranker with default alpha (0.7 tfidf, 0.3 skills)
ranker = ResumeRanker(alpha=0.7)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize DB
with app.app_context():
    db.create_all()
    print("✅ Database initialized successfully!")
    print(f"📁 Database location: {database_path}")

@app.route('/')
def index():
    jobs = Job.query.all()
    stats = {'total_resumes': Resume.query.count(), 'total_jobs': Job.query.count()}
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
            # remove file to keep uploads clean
            try:
                os.remove(filepath)
            except OSError:
                pass

            if parsed_data:
                # Normalize extracted data
                name = parsed_data.get('name') if isinstance(parsed_data, dict) else ''
                email = parsed_data.get('email') if isinstance(parsed_data, dict) else ''
                phone = parsed_data.get('phone') if isinstance(parsed_data, dict) else ''
                skills = parsed_data.get('skills') if isinstance(parsed_data, dict) else ''
                education = parsed_data.get('education') if isinstance(parsed_data, dict) else ''
                experience = parsed_data.get('experience') if isinstance(parsed_data, dict) else ''
                raw_text = parsed_data.get('raw_text') if isinstance(parsed_data, dict) else ''

                # Convert skills list to comma string for storage (if parser returns list)
                if isinstance(skills, (list, tuple)):
                    skills_str = ','.join([s.strip() for s in skills if s and str(s).strip()])
                elif isinstance(skills, str):
                    skills_str = skills.strip()
                else:
                    skills_str = ''

                resume = Resume(
                    name=name or '',
                    email=email or '',
                    phone=phone or '',
                    skills=skills_str,
                    education=education or '',
                    experience=experience or '',
                    raw_text=raw_text or ''
                )
                db.session.add(resume)
                uploaded_count += 1

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

    # Prepare resume data for ranker
    resume_data = []
    for r in resumes:
        # prepare skills_list: if skills stored as comma string -> list
        skl = getattr(r, 'skills', '') or ''
        if isinstance(skl, str):
            skl_list = [s.strip() for s in skl.split(',') if s.strip()]
        elif isinstance(skl, (list, tuple)):
            skl_list = skl
        else:
            skl_list = []

        resume_data.append({
            'id': r.id,
            'raw_text': r.raw_text or '',
            'skills_list': skl_list
        })

    # Get ranked results from ranker
    ranked = ranker.rank_resumes(resume_data, job.description or '')

    # Clear previous MatchScore entries for this job
    MatchScore.query.filter_by(job_id=job_id).delete()

    # Save new match scores. MatchScore model assumed to have fields (resume_id, job_id, score)
    for item in ranked:
        rid, final_score, meta = item
        # convert to 0..1 stored numeric (we keep final_score in 0..1)
        try:
            score_val = float(final_score)
        except Exception:
            score_val = 0.0
        match = MatchScore(resume_id=rid, job_id=job_id, score=score_val)
        # If you'd like to persist meta, add a column to MatchScore (e.g., meta JSON)
        # e.g., match.meta = json.dumps(meta)
        db.session.add(match)

    db.session.commit()
    flash(f'Successfully ranked {len(resumes)} resumes for \"{job.title}\"!', 'success')
    return redirect(url_for('results', job_id=job_id))

@app.route('/results/<int:job_id>')
def results(job_id):
    job = Job.query.get_or_404(job_id)
    matches = MatchScore.query.filter_by(job_id=job_id).order_by(MatchScore.score.desc()).all()

    results_list = []
    for match in matches:
        resume = Resume.query.get(match.resume_id)
        if not resume:
            continue

        skills_raw = getattr(resume, 'skills', '') or ''
        if isinstance(skills_raw, (list, tuple)):
            skills_text = ','.join([s for s in skills_raw if s])
            skills_list = [s for s in skills_raw if s]
        else:
            skills_text = str(skills_raw)
            skills_list = [s.strip() for s in skills_text.split(',') if s.strip()]

        results_list.append({
            'resume': resume,
            # convert stored score (0..1) to percentage for display
            'score': round((match.score or 0.0) * 100, 2),
            'match': match,
            'skills_text': skills_text,
            'skills_list': skills_list,
            'skill_count': len(skills_list)
        })

    return render_template('results.html', job=job, results=results_list)

@app.route('/dashboard')
def dashboard():
    resumes = Resume.query.order_by(Resume.uploaded_at.desc()).all()
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    stats = {'total_resumes': Resume.query.count(), 'total_jobs': Job.query.count()}
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
