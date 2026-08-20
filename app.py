import streamlit as st
import re
import PyPDF2
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ATS Resume Matcher", page_icon="🎯", layout="wide")

# Download stop words once
nltk.download('stopwords', quiet=True)

# 1. Extraction Logic
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# 2. NLP Preprocessing Logic 
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {
        'experience', 'seeking', 'candidate', 'ideal', 'strong', 
        'foundation', 'required', 'familiarity', 'huge', 'plus', 
        'like', 'tools', 'technologies', 'web', 'software', 'engineer'
    }
    all_stopwords = stop_words.union(custom_stopwords)
    return ' '.join([w for w in text.split() if w not in all_stopwords])

# 3. AI Matcher Logic
def get_ats_score(resume_text, job_desc):
    cleaned_resume = preprocess(resume_text)
    cleaned_jd = preprocess(job_desc)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return round(similarity[0][0] * 100, 2)

# --- ENHANCED WEB INTERFACE (FRONTEND) ---

# Main Header
st.title("🎯 AI-Powered ATS Resume Matcher")
st.markdown("A lightweight NLP pipeline to evaluate candidate alignment using TF-IDF and Cosine Similarity.")

# SIDEBAR: Keep inputs tucked away for a cleaner UI
with st.sidebar:
    st.header("📄 Data Inputs")
    uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type="pdf")
    job_description = st.text_area("2. Paste Job Description:", height=250)
    analyze_button = st.button("Analyze Match", type="primary", use_container_width=True)

# MAIN AREA: Results Dashboard
if analyze_button:
    if uploaded_file is not None and job_description:
        with st.spinner("Initializing NLP pipeline & calculating vectors..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Edge Case Safeguard
            if not resume_text.strip():
                st.error("Error: Could not extract text. Please ensure this is a text-based PDF.")
            else:
                score = get_ats_score(resume_text, job_description)
                
                # Keyword Analysis Logic
                clean_jd_words = set(preprocess(job_description).split())
                clean_res_words = set(preprocess(resume_text).split())
                
                matched_words = clean_jd_words.intersection(clean_res_words)
                missing_words = clean_jd_words.difference(clean_res_words)
                
                st.markdown("---")
                
                # Top Metric Section
                st.subheader("📊 Match Results")
                st.metric(label="ATS Cosine Similarity Score", value=f"{score}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Two-Column Keyword Analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ Matched Skills ({len(matched_words)})")
                    for word in matched_words:
                        st.write(f"- {word.capitalize()}")
                        
                with col2:
                    st.error(f"❌ Missing Skills ({len(missing_words)})")
                    for word in missing_words:
                        st.write(f"- {word.capitalize()}")
                
                st.markdown("---")
                
                # Explainability Feature (Great for Interviews)
                with st.expander("🔍 Behind the Scenes: View Raw Extracted Text"):
                    st.write("This is the raw text the engine extracted from the PDF before NLP cleaning:")
                    st.text(resume_text)
                    
    else:
        st.warning("👈 Please upload a resume and paste a job description in the sidebar to begin.")
else:
    # Empty State Dashboard
    st.info("👈 Waiting for data. Upload a resume and paste a job description in the sidebar to test the engine.")