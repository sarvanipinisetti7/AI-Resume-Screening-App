import os
from pypdf import PdfReader
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")

# NLP PREPROCESSING

def preprocess_text(raw_text):

    lowercased_text = raw_text.lower()

    cleaned_text = re.sub(r'[^a-zA-Z ]', '', lowercased_text)

    all_words = word_tokenize(cleaned_text)

    english_stop_words = set(stopwords.words("english"))

    meaningful_words = []

    for word in all_words:
        if word not in english_stop_words:
            meaningful_words.append(word)

    return " ".join(meaningful_words)

#  PDF EXTRACTION

def extract_text(pdf_file_path):

    extracted_text = ""

    pdf_reader = PdfReader(pdf_file_path)

    for page in pdf_reader.pages:
        extracted_text += page.extract_text()

    return extracted_text

#JOB DESCRIPTION 

with open("job_description.txt", "r") as job_file:
    job_description_text = job_file.read()

print("\nJob Description:")
print("--------------------")
print(job_description_text)

# REQUIRED SKILLS

required_skill_list = [

    "python",
    "sql",
    "excel",
    "pandas",
    "numpy",
    "machine learning",
    "power bi",
    "data analysis",
    "data visualization"

]

resumes_folder_path = "resumes"

all_candidates = []

#  PROCESS RESUMES 

for resume_file_name in os.listdir(resumes_folder_path):

    if resume_file_name.endswith(".pdf"):

        resume_file_path = os.path.join(resumes_folder_path, resume_file_name)

        raw_resume_text = extract_text(resume_file_path)

        print("\nResume Name:", resume_file_name)
        print("--------------------")
        print("Extracted length:", len(raw_resume_text))

        cleaned_resume_text = preprocess_text(raw_resume_text)

        cleaned_job_description = preprocess_text(job_description_text)


        # TF-IDF Similarity

        documents_to_compare = [

            cleaned_resume_text,
            cleaned_job_description

        ]

        tfidf_vectorizer = TfidfVectorizer()

        tfidf_matrix = tfidf_vectorizer.fit_transform(documents_to_compare)

        similarity_matrix = cosine_similarity(

            tfidf_matrix[0:1],
            tfidf_matrix[1:2]

        )

        tfidf_similarity_score = similarity_matrix[0][0] * 100


        # Skill Matching

        matched_skill_list = []

        for skill in required_skill_list:

            if skill in raw_resume_text.lower():

                matched_skill_list.append(skill)


        skill_match_percentage = (

            len(matched_skill_list) /
            len(required_skill_list)

        ) * 100


        print("\nSkill Match Analysis")
        print("--------------------")

        for matched_skill in matched_skill_list:

            print(matched_skill, "✅")


        print(
            "\nMatched Skills:",
            len(matched_skill_list),
            "/",
            len(required_skill_list)
        )

        print(
            "Skill Match Percentage:",
            round(skill_match_percentage, 2),
            "%"
        )


        # FINAL ATS SCORE

        experience_match_score = 80

        project_match_score = 85


        final_ats_score = (

            skill_match_percentage * 0.5 +

            tfidf_similarity_score * 0.2 +

            experience_match_score * 0.15 +

            project_match_score * 0.15

        )


        all_candidates.append({

            "resume": resume_file_name,

            "score": round(final_ats_score, 2)

        })

#RANKING 

ranked_candidates = sorted(

    all_candidates,

    key=lambda candidate: candidate["score"],

    reverse=True

)

print("\nCandidate Ranking")
print("--------------------")

current_rank = 1

for candidate in ranked_candidates:

    print(

        "Rank",
        current_rank,
        ":",
        candidate["resume"],
        "-",
        candidate["score"],
        "%"

    )

    current_rank += 1

# RECOMMENDATION

top_candidate_score = ranked_candidates[0]["score"]

print("\nRecommendation:")
print("--------------------")

if top_candidate_score >= 70:

    print(
        "Suitable Candidate for Data Analyst Role"
    )

else:

    print(
        "Needs More Skill Alignment"
    )