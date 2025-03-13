from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import os
import zipfile
import pdfplumber
import shutil
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# Load Models
t5_model_name = "t5-base"
sbert_model_name = "sentence-transformers/msmarco-distilbert-base-v4"  # Using a ranking-focused model

t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

sbert_model = SentenceTransformer(sbert_model_name)

app = Flask(__name__, static_url_path='/static')
app.secret_key = "nn4w89u4hg89hf"

UPLOAD_FOLDER = 'uploads'
EXTRACT_FOLDER = 'extracted_resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

global_job_description = ""
ranks = []

WEIGHT_MAP = {
    "must have": 1.0, "required": 1.0, "mandatory": 1.0,
    "preferred": 0.7, "nice to have": 0.5, "optional": 0.3
}


def preprocess_text(text):
    """Cleans text for processing."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def extract_requirements_t5(job_description):
    """Forces T5 to extract **skills** instead of full sentences."""
    prompt = f"Extract short key skills and requirements from this job description: {job_description}"
    input_ids = t5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).input_ids
    output_ids = t5_model.generate(input_ids, max_length=50, num_return_sequences=1)

    extracted_text = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    extracted_requirements = re.split(r",|\n", extracted_text)  # Split by commas/newlines
    extracted_requirements = [req.strip().lower() for req in extracted_requirements if req.strip()]

    # Remove long sentences, keep short skills
    extracted_requirements = [req for req in extracted_requirements if len(req.split()) <= 3]

    return assign_weights(job_description, extracted_requirements)


def assign_weights(job_description, extracted_requirements):
    """Assigns weight to each requirement based on job description context."""
    job_description = job_description.lower()
    weights = {}

    for req in extracted_requirements:
        weight = 0.5  # Default weight
        for phrase, w in WEIGHT_MAP.items():
            if phrase in job_description and req in job_description:
                weight = max(weight, w)

        weights[req] = round(weight, 2)

    total_weight = sum(weights.values()) or 1
    weights = {k: round(v / total_weight, 2) for k, v in weights.items()}

    return weights


def sbert_match_score(weighted_requirements, resume_text):
    """Compares the job requirements with the resume using SBERT similarity scoring."""
    resume_sentences = resume_text.split(".")  # Compare sentence by sentence

    match_score = 0
    total_weight = sum(weighted_requirements.values()) or 1  # Prevent division by zero

    matched_requirements = []
    missed_requirements = []

    for requirement, weight in weighted_requirements.items():
        req_embedding = sbert_model.encode(requirement, convert_to_tensor=True)

        max_similarity = 0
        for sentence in resume_sentences:
            sentence_embedding = sbert_model.encode(sentence.strip(), convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(req_embedding, sentence_embedding).item()
            max_similarity = max(max_similarity, similarity)

        if max_similarity > 0.3:  # Lowered threshold for better matching
            match_score += max_similarity * weight
            matched_requirements.append(requirement)
        else:
            missed_requirements.append(requirement)

    normalized_score = round(match_score / total_weight, 2)  # Normalize score

    return normalized_score, matched_requirements, missed_requirements


def process_resume(job_description, resume_text):
    """Processes a single resume against job description and calculates ranking score."""
    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)

    weighted_requirements = extract_requirements_t5(job_description)
    match_score, matched_reqs, missed_reqs = sbert_match_score(weighted_requirements, resume_text)

    activities_keywords = ["internship", "project", "research", "club", "society", "leadership"]
    activities_count = sum(resume_text.count(word) for word in activities_keywords)

    final_score = (match_score * 0.7) + (activities_count * 0.3)

    explanation = {
        "Matched Requirements": matched_reqs,
        "Missed Requirements": missed_reqs,
        "Job Match Score": round(match_score * 100, 2),
        "Activity Count": activities_count,
        "Final Weighted Score": round(final_score * 100, 2),
        "Ranking Justification": f"Matched {len(matched_reqs)} out of {len(matched_reqs) + len(missed_reqs)} requirements. "
                                 f"Includes {activities_count} relevant activities."
    }

    return round(final_score * 100, 2), explanation


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

    if request.method == 'POST':
        data = request.json
        updated_criteria = {item['name']: item['weight'] for item in data['criteria']}
        session['updated_criteria'] = updated_criteria
        return jsonify({"redirect": url_for('upload_resumes')})

    # Extract requirements if not already in session
    if 'updated_criteria' not in session:
        extracted_requirements = extract_requirements_t5(global_job_description)
        print("Extracted Requirements:", extracted_requirements)  # Debugging print
        if not extracted_requirements:
            print("Warning: No extracted requirements found!")

        session['updated_criteria'] = extracted_requirements

    return render_template(
        'criteria.html',
        criteria={'criteria': [{'name': k, 'weight': v} for k, v in session['updated_criteria'].items()]}
    )

@app.route('/upload_resumes', methods=['GET', 'POST'])
def upload_resumes():
    """Handles resume ZIP upload and extraction."""
    if request.method == 'POST':
        if 'resumeZip' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['resumeZip']
        shutil.rmtree(EXTRACT_FOLDER)
        os.makedirs(EXTRACT_FOLDER)

        if not file.filename.endswith('.zip'):
            return jsonify({"error": "Only ZIP files are allowed"}), 400

        zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)

        filenames = []
        extracted_files = []
        for root, _, files in os.walk(EXTRACT_FOLDER):
            for filename in files:
                filenames.append(filename)
                file_path = os.path.join(root, filename)
                extracted_files.append(file_path)

        global ranks
        ranks = []

        # Use user-modified criteria from session
        weighted_requirements = session.get('updated_criteria', {})

        for i, pdf_file in enumerate(extracted_files):
            if pdf_file.endswith('.pdf'):
                pdf_text = extract_text_from_pdf(pdf_file)
                match_score, matched_reqs, missed_reqs = sbert_match_score(weighted_requirements, pdf_text)
                activities_keywords = ["internship", "project", "research", "club", "society", "leadership"]
                activities_count = sum(pdf_text.count(word) for word in activities_keywords)

                final_score = (match_score * 0.7) + (activities_count * 0.3)

                explanation = {
                    "Matched Requirements": matched_reqs,
                    "Missed Requirements": missed_reqs,
                    "Job Match Score": round(match_score * 100, 2),
                    "Activity Count": activities_count,
                    "Final Weighted Score": round(final_score * 100, 2),
                    "Ranking Justification": f"Matched {len(matched_reqs)} out of {len(matched_reqs) + len(missed_reqs)} requirements. "
                                             f"Includes {activities_count} relevant activities."
                }

                ranks.append({
                    "filename": filenames[i],
                    "score": round(final_score * 100, 2),
                    "explanation": explanation
                })

        ranks = sorted(ranks, key=lambda x: x["score"], reverse=True)
        session['ranks'] = ranks
        return redirect(url_for('results'))

    return render_template('resume.html')



@app.route('/results', methods=['GET'])
def results():
    """Displays ranking results."""
    ranks = session.get('ranks', [])
    return render_template('results.html', ranks=ranks)


if __name__ == '__main__':
    app.run()
