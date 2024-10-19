from flask import Flask, request, render_template
import nest_asyncio
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import json
from pymongo import MongoClient

app = Flask(__name__, static_url_path='/static')
nest_asyncio.apply()

GOOGLE_API_KEY = "*******"
model = genai.GenerativeModel('gemini-1.5-flash')
genai.configure(api_key = GOOGLE_API_KEY)

# Database Credentials
client = MongoClient("localhost", 27017)
db = client["ResumeDataset"]
collection = db["collection"]
data = collection.find()

# Get resumes from database
resume_info = []
register = {}
for x in data:
    resume_info.append(x['Resume'])
    register[x['Resume']] = x['RegNo']

# Apply weights to embeddings
def apply_weightage(embedding, text, keyword_weights):
    weighted_embedding = np.copy(embedding) 
    for keyword, weight in keyword_weights.items():
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text.lower()):
            weighted_embedding += weight * embedding 
    return weighted_embedding

def preprocess_text(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def process(job, resume_info):
    prompt = f"""
    Analyze the following job description and suggest a weighting system for the following features:
    1.	Internship
    2.	Hackathon
    3.	Coding platforms
    4.	Industry certificate
    5.	NPTEL/Coursera courses
    6.	Github repository
    7.	Real time Project experience
    8.	CGPA and no of arrears, minor degree
    9.  10th and 12th grade score
    10.	Research experience/project and Industry sponsored project

    Job Description:
    {job}

    Preferences in job description should be given high points
    Please provide a weight (between 0 and 1) for each of the above features, with a higher weight for more important features based on the job description.
    Please dont give any explaination.
    I want it in json format
    """

    # Use the text generation API to get the weightage suggestions
    response = model.generate_content(prompt)
    res = response.text
    result = res[7:-3]
    keyword_weights = json.loads(result)

    resumes = [preprocess_text(r) for r in resume_info]
    job = preprocess_text(job)
    resume_embeddings = [genai.embed_content(model="models/text-embedding-004", task_type="SEMANTIC_SIMILARITY", content=resume)["embedding"] for resume in resumes]
    job_embedding = genai.embed_content(model="models/text-embedding-004", task_type="SEMANTIC_SIMILARITY", content=job)["embedding"]

    job_embedding = np.array(job_embedding)
    resume_embeddings = np.array(resume_embeddings)

    weighted_job_embedding = apply_weightage(job_embedding, job, keyword_weights)
    weighted_resume_embeddings = [apply_weightage(embedding, resume, keyword_weights) for embedding, resume in
                                  zip(resume_embeddings, resumes)]

    ranks = {}
    similarities = cosine_similarity([weighted_job_embedding], weighted_resume_embeddings)
    ranked_resumes = np.argsort(similarities[0])[::-1]

    for idx in ranked_resumes:
        ranks[register[resume_info[idx]]] = similarities[0][idx]
    return ranks


@app.route('/', methods=['GET', 'POST'])
def home():
    ranks = {}
    if request.method == 'POST':
        job_description = request.form['job_description']
        ranks = process(job_description, resume_info)
    return render_template('index.html', ranks=ranks)


if __name__ == '__main__':
    app.run()
