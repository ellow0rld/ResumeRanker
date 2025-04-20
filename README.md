# ResumeRanker

It's a website built using Flask, to rank student's resume based on the job description. The project uses the following parameters to evaluate the similarities between job description and resumes.

Weights are assigned to each of the parameters based on the job description, with the help of Gemini Model.
The job description and resumes are then embedded into numerical vectors, based on semantic similarity and the above parameters.
Vector Search is performed to rank the resumes based on cosine similarity scores.

# Output
