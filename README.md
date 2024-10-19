# ResumeRanker

It's a website built using Flask, to rank student's resume based on the job description. The project uses the following parameters to evaluate the similarities between job description and resumes.

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

Weights are assigned to each of the parameters based on the job description, with the help of Gemini Model.
The job description and resumes are then embedded into numerical vectors, based on semantic similarity and the above parameters.
Vector Search is performed to rank the resumes based on cosine similarity scores.

# Output

The website
![Screenshot 2024-10-20 050133](https://github.com/user-attachments/assets/452145c7-f27e-4807-a455-523fec0ccb48)

Job Description
![Screenshot 2024-10-20 050153](https://github.com/user-attachments/assets/84e4f797-b39d-4b2f-85b0-71db9939e640)

Output
![Screenshot 2024-10-20 050211](https://github.com/user-attachments/assets/8716fcf9-40a8-4e95-8139-558f514ba694)

Weights generated for the given job description
![Screenshot 2024-10-20 050231](https://github.com/user-attachments/assets/1b49e1f1-7822-4e3e-b508-5ac1d3f3e207)

Resume Database
![Screenshot 2024-10-20 050446](https://github.com/user-attachments/assets/e1a95813-9c3e-4c02-bed5-757f3e1ecaf4)
