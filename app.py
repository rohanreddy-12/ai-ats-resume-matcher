import streamlit as st
import re
import PyPDF2
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ATS Resume Matcher", page_icon="✨", layout="wide")

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

# 3. AI Matcher Logic (With Commercial Normalization)
def get_ats_score(resume_text, job_desc):
    cleaned_resume = preprocess(resume_text)
    cleaned_jd = preprocess(job_desc)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
    similarity = cosine_similarity(vectors[0], vectors[1])
    
    # Normalize score to a 0-100 scale
    raw_score = similarity[0][0] * 100
    adjusted_score = min(100.00, raw_score * 3) 
    return round(adjusted_score, 2)

# --- CUSTOM CSS FOR UI ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<p class="main-header">✨ AI-Powered ATS Matcher</p>', unsafe_allow_html=True)
st.markdown("**A sleek NLP pipeline to evaluate candidate alignment using TF-IDF & Cosine Similarity.**")
st.divider()

# SIDEBAR
with st.sidebar:
    st.header("📄 Upload Data")
    st.caption("Feed the AI your resume and the target job description.")
    uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type="pdf")
    job_description = st.text_area("2. Paste Job Description:", height=250)
    analyze_button = st.button("🚀 Analyze Match", type="primary", use_container_width=True)

# MAIN AREA
if analyze_button:
    if uploaded_file is not None and job_description:
        with st.spinner("Initializing NLP pipeline & calculating vectors..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if not resume_text.strip():
                st.error("❌ Error: Could not extract text. Please ensure this is a text-based PDF.")
            else:
                score = get_ats_score(resume_text, job_description)
                clean_jd_words = set(preprocess(job_description).split())
                clean_res_words = set(preprocess(resume_text).split())
                
                matched_words = clean_jd_words.intersection(clean_res_words)
                missing_words = clean_jd_words.difference(clean_res_words)
                
                # --- DYNAMIC RESULTS DASHBOARD ---
                st.subheader("📊 Match Results")
                
                # Score Metric & Visual Progress Bar
                st.metric(label="ATS Match Score", value=f"{score}%")
                # Convert percentage out of 100 to a decimal between 0.0 and 1.0 for the progress bar
                st.progress(min(score / 100, 1.0))
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Two-Column Keyword Analysis (Comma-separated for cleaner look)
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ **Matched Skills ({len(matched_words)})**")
                    if matched_words:
                        st.write(", ".join([word.capitalize() for word in matched_words]))
                    else:
                        st.write("*None found.*")
                        
                with col2:
                    st.error(f"❌ **Missing Skills ({len(missing_words)})**")
                    if missing_words:
                        st.write(", ".join([word.capitalize() for word in missing_words]))
                    else:
                        st.write("*None missing!*")
                
                st.markdown("---")
                with st.expander("🔍 Behind the Scenes: View Raw Extracted Text"):
                    st.text(resume_text)
                    
    else:
        st.warning("👈 Please upload a resume and paste a job description in the sidebar to begin.")
else:
    # --- EMPTY STATE / LANDING PAGE ---
    st.info("👋 Welcome! Upload a resume and paste a job description in the sidebar to test the engine.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1️⃣ Data Ingestion")
        st.write("Upload any PDF resume. Our backend extracts the raw text using the `PyPDF2` library.")
        
    with col2:
        st.subheader("2️⃣ NLP Processing")
        st.write("The text is sanitized using Regular Expressions and `NLTK` to remove stopwords and non-technical noise.")
        
    with col3:
        st.subheader("3️⃣ Vector Math")
        st.write("A `TF-IDF Vectorizer` converts the text into numeric matrices, and `Cosine Similarity` calculates the mathematical match.")