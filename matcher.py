import re
import PyPDF2
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stop words once
nltk.download('stopwords', quiet=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])

def get_ats_score(resume_text, job_desc):
    cleaned_resume = preprocess(resume_text)
    cleaned_jd = preprocess(job_desc)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return round(similarity[0][0] * 100, 2)

if __name__ == "__main__":
    # Test job description
    sample_jd = """
We are seeking a Software Engineer with a strong foundation in Data Structures and Algorithms. 
The ideal candidate should have experience in Java, Python, and JavaScript. 
Experience with Web Technologies like HTML5, CSS3, and DOM Manipulation is required. 
Familiarity with serverless architectures, REST APIs, and tools like Git and VS Code is a huge plus.
"""
    
    # Extract, score, and print
    resume_content = extract_text_from_pdf("CRM.pdf")
    score = get_ats_score(resume_content, sample_jd)
    print(f"ATS Match Score: {score}%")