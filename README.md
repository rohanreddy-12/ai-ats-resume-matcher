# 🎯 AI-Powered ATS Resume Matcher

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK-green.svg)

**Live Demo:** [View the Web App Here](https://ai-ats-resume-matcher-rohan.streamlit.app/)

A lightweight Natural Language Processing (NLP) SaaS pipeline built to evaluate how well a candidate's resume aligns with a specific job description. This tool simulates how commercial Applicant Tracking Systems (ATS) filter candidates using vector mathematics and keyword extraction.

---

## 🚀 Features

* **PDF Data Ingestion:** Uses `PyPDF2` to extract raw text data directly from uploaded resume files.
* **NLP Preprocessing:** Utilizes Regular Expressions and `NLTK` to sanitize text, standardize casing, and remove standard English stopwords alongside custom "resume fluff" (e.g., "synergy", "candidate", "seeking").
* **Vector Mathematics:** Implements `scikit-learn`'s **TF-IDF Vectorizer** to convert text into numeric matrices, weighing the mathematical importance of technical terms.
* **Similarity Scoring:** Uses **Cosine Similarity** to calculate the exact angle of distance between the resume vector and the job description vector, normalized to a 0-100% human-readable scale.
* **Keyword Analysis Engine:** Applies Python Set Theory (Intersections and Differences) to instantly isolate matched technical skills and highlight missing requirements.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Frontend UI / Hosting:** Streamlit
* **Machine Learning:** scikit-learn (TF-IDF, Cosine Similarity)
* **Natural Language Processing:** NLTK, Regular Expressions (re)
* **Document Parsing:** PyPDF2

---

## 💻 Run Locally

If you want to clone this repository and run the engine on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/rohanreddy-12/ai-ats-resume-matcher.git](https://github.com/rohanreddy-12/ai-ats-resume-matcher.git)
cd ai-ats-resume-matcher