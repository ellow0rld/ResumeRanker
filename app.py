from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_file
import os
import zipfile
import pdfplumber
import shutil
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import json

# Configure Gemini API
genai.configure(api_key="AIzaSyDRzUidjMLSGoWok7ooNNZHbciMpgizqC0")
model = genai.GenerativeModel("gemini-2.0-flash")
# Load Sentence Transformers for similarity scoring
sbert_model = SentenceTransformer("all-mpnet-base-v2")

app = Flask(__name__, static_url_path='/static')
app.secret_key = "nn4w89u4hg89hf"

UPLOAD_FOLDER = 'uploads'
EXTRACT_FOLDER = 'extracted_resumes'
HIGHLIGHTED_FOLDER = 'highlighted_resumes'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)
os.makedirs(HIGHLIGHTED_FOLDER, exist_ok=True)

global_job_description = ""
ranks = []

# Hyperparameter for score weighting
ALPHA = 0.4   # Weight for skill match score vs cosine similarity


def preprocess_text(text):
    """Cleans text for processing."""
    return re.sub(r'\s+', ' ', text.replace("\n", " ").replace("\r", " ")).strip().lower()


def extract_requirements_gemini(job_description):
    """Uses Gemini 2.0 Flash to extract key requirements with weights."""
    prompt = f"""
    Analyze this Job description and generate top 5 important keyword with weightage depending on preferences and required, as 1-D json.
    The weights must sum up to 100.
    Focus on the technical skills
    Job Description:
    {job_description}
    json format: [criteria1:weight1,criteria2:weight2,...]
    """
    response = model.generate_content(prompt)
    res = response.text
    #print(res)
    result = res[7:-4]
    keyword_weights = json.loads(result)
    try:
        return keyword_weights
    except Exception as e:
        print("Error parsing Gemini output:", e)
        return {}

def extract_summary_gemini(job_description):
    prompt = f'''summarize the requirements of this job description. Plain text.
    {job_description}'''
    response = model.generate_content(prompt)
    summary = response.text
    return summary

def sbert_match_score(weighted_requirements, resume_text, summary):
    """Computes the weighted semantic similarity between job requirements and a resume."""
    if not weighted_requirements or not resume_text.strip():
        return 0.0

    req_texts, weights = zip(*weighted_requirements.items())
    weights = np.array(weights)
    # Encode job requirements and resume text
    sum_embeddings = sbert_model.encode(summary, convert_to_tensor=True)
    req_embeddings = sbert_model.encode(req_texts, convert_to_tensor=True)
    resume_embedding = sbert_model.encode(resume_text, convert_to_tensor=True)
    #print(summary)
    #print("summary: ", sum_embeddings)
    #print("requirement: ", req_embeddings)
    #print("resume: ", resume_embedding)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(sum_embeddings, resume_embedding).squeeze().cpu().numpy()*100
    #print("Summary: ", similarity_score)
    # Weighted match score
    similarities = util.pytorch_cos_sim(req_embeddings, resume_embedding).squeeze().cpu().numpy()
    match_score = (np.dot(similarities, weights) / np.sum(weights))*100
    #print("Match Score: ", similarities, match_score)
    scores = {'Context based Score' : round(similarity_score, 2), 'Keyword based Score': round(match_score, 2)}
    # Compute overall score with alpha blending
    return [round((ALPHA * match_score + (1 - ALPHA) * similarity_score.mean()), 2), scores]


def highlight_keywords_in_pdf(pdf_path, output_path, keywords):
    """Highlights matched keywords in the given PDF."""
    doc = fitz.open(pdf_path)

    for page in doc:
        for keyword in keywords:
            # Split keyword by common delimiters like '/', ',', and space
            sub_keywords = re.split(r'[/,\s]+', keyword)

            for sub_keyword in sub_keywords:
                if sub_keyword.strip():  # Avoid empty strings
                    for inst in page.search_for(sub_keyword):
                        page.add_highlight_annot(inst)

    doc.save(output_path)
    doc.close()


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


@app.route('/', methods=['GET', 'POST'])
def home():
    global global_job_description
    if request.method == 'POST':
        global_job_description = request.form['job_description']
        return redirect(url_for('criteria'))
    return render_template('index.html')


@app.route('/criteria', methods=['GET', 'POST'])
def criteria():
    global global_job_description

    # Extract requirements using Gemini
    criteria_to_display = extract_requirements_gemini(global_job_description)

    # Store extracted criteria in session
    session['extracted_criteria'] = criteria_to_display

    if request.method == 'POST':
        data = request.json
        updated_criteria = {item['name']: item['weight'] for item in data['criteria']}
        session['updated_criteria'] = updated_criteria
        return jsonify({"redirect": url_for('upload_resumes')})

    return render_template(
        'criteria.html',
        criteria={'criteria': [{'name': k, 'weight': v} for k, v in criteria_to_display.items()]}
    )


@app.route('/upload_resumes', methods=['GET', 'POST'])
def upload_resumes():
    if request.method == 'POST':
        if 'resumeZip' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['resumeZip']
        shutil.rmtree(EXTRACT_FOLDER, ignore_errors=True)
        os.makedirs(EXTRACT_FOLDER)

        if not file.filename.endswith('.zip'):
            return jsonify({"error": "Only ZIP files are allowed"}), 400

        zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)

        filenames, extracted_files = [], []
        for root, _, files in os.walk(EXTRACT_FOLDER):
            for filename in files:
                filenames.append(filename)
                extracted_files.append(os.path.join(root, filename))

        global ranks
        ranks = []
        weighted_requirements = session.get('updated_criteria', {})

        for i, pdf_file in enumerate(extracted_files):
            if pdf_file.endswith('.pdf'):
                pdf_text = extract_text_from_pdf(pdf_file)
                summary = extract_summary_gemini(global_job_description)
                return_val = sbert_match_score(weighted_requirements, pdf_text, summary)
                score = return_val[0]
                scores = return_val[1]
                print(filenames[i])
                print(scores)
                highlighted_pdf_path = os.path.join(EXTRACT_FOLDER, f"highlighted_{filenames[i]}")
                highlight_keywords_in_pdf(pdf_file, highlighted_pdf_path, weighted_requirements.keys())
                final_score = round(score, 2)
                if final_score >= 10:
                    ranks.append({
                        "filename": filenames[i],
                        "score": final_score,
                        "highlighted_link": url_for('serve_highlighted_resume', filename=f"highlighted_{filenames[i]}"),
                        "explanation": scores
                    })

        ranks = sorted(ranks, key=lambda x: x["score"], reverse=True)
        session['ranks'] = ranks
        return redirect(url_for('results'))

    return render_template('resume.html')


@app.route('/highlighted/<filename>')
def serve_highlighted_resume(filename):
    return send_file(os.path.join(EXTRACT_FOLDER, filename), as_attachment=True)


@app.route('/results')
def results():
    return render_template('results.html', ranks=session.get('ranks', []))


if __name__ == '__main__':
    app.run()
