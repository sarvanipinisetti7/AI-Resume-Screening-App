import streamlit as st
from pypdf import PdfReader
import nltk
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# NLTK setup

try:
    nltk.data.find("tokenizers/punkt_tab")
except:
    nltk.download("punkt_tab")


try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")


try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

st.title("🤖 AI Resume Screening & Candidate Ranking System")

# Extract PDF text

def extract_text(pdf):

    text = ""

    reader = PdfReader(pdf)

    for page in reader.pages:

        text += page.extract_text()

    return text

# NLP preprocessing

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        '',
        text
    )

    words = word_tokenize(text)

    stop_words = set(
        stopwords.words("english")
    )

    words = [

        word for word in words

        if word not in stop_words

    ]

    return " ".join(words)

# Upload resumes

resumes = st.file_uploader(

    "Upload Multiple Resume PDFs",

    type=["pdf"],

    accept_multiple_files=True

)

# JD input

job_description = st.text_area(

    "Enter Job Description"

)

required_skills = [

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

if st.button("Analyze Resumes"):

    if resumes and job_description:

        candidates = []

        clean_jd = preprocess_text(
            job_description
        )

        for resume in resumes:

            resume_text = extract_text(
                resume
            )

            clean_resume = preprocess_text(
                resume_text
            )

            # Similarity

            vectorizer = TfidfVectorizer()

            matrix = vectorizer.fit_transform(

                [
                    clean_resume,
                    clean_jd
                ]

            )

            similarity = cosine_similarity(

                matrix[0:1],
                matrix[1:2]

            )

            similarity_score = (

                similarity[0][0] * 100

            )

            # Skill matching

            matched = []

            for skill in required_skills:

                if skill in resume_text.lower():

                    matched.append(skill)

            missing = []

            for skill in required_skills:

                if skill not in matched:

                    missing.append(skill)

            skill_score = (

                len(matched)
                /
                len(required_skills)

            ) * 100

            final_score = (

                skill_score * 0.5
                +
                similarity_score * 0.5

            )

            candidates.append(

                {
                "resume"  : resume.name,
                "score"   : final_score,
                "matched" : matched,
                "missing" : missing
                }

            )

        # Ranking

        candidates = sorted(

            candidates,

            key=lambda x: x["score"],

            reverse=True

        )

        st.success("Analysis Completed ✅")

        # Ranking table

        st.subheader("🏆 Candidate Ranking")

        table = []

        rank = 1

        for c in candidates:

            table.append(

                {
                "Rank"      : rank,
                "Resume"    : c["resume"],
                "Score (%)" : f"{c['score']:.2f}"
                }

            )

            rank += 1

        df = pd.DataFrame(table)

        st.table(df)

        # ✅ NEW — Final Recommendation (ikkada add chesamu)

        st.subheader("📋 Final Recommendation")

        if candidates[0]["score"] >= 70:

            st.success(
                f"✅ {candidates[0]['resume']} - Suitable Candidate for Data Analyst Role!"
            )

        else:

            st.warning(
                "⚠️ No candidate meets the required threshold!"
            )

        # Skill details

        st.subheader("Skill Analysis")

        for c in candidates:

            st.write("###", c["resume"])

            st.write(
                "Score:",
                f"{c['score']:.2f}",
                "%"
            )

            st.write("Matched Skills ✅")

            for skill in c["matched"]:

                st.write(skill, "✅")

            st.write("Missing Skills ❌")

            if len(c["missing"]) == 0:

                st.write("No missing skills ✅")

            else:

                for skill in c["missing"]:

                    st.write(skill, "❌")

            st.divider()

    else:

        st.warning(
            "Upload resumes and enter Job Description"
        )