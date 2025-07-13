import streamlit as st
import pickle, re, os, ssl
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK data directory
nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_path, exist_ok=True)

if nltk_path not in nltk.data.path:
    nltk.data.path.append(nltk_path)

try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

for pkg in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_path)

# Load models
tfid = pickle.load(open("tfid.pkl", "rb"))
clf = pickle.load(open("clf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def clean_resume(text: str) -> str:
    text = re.sub(r"https?\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"[!#$%&'()*+,<=>?@\[\]^_`{|}~]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def highlight_keywords(resume_text: str, key_list):
    tokens = word_tokenize(resume_text.lower())
    sw = set(stopwords.words("english"))
    return list({w for w in tokens if w in key_list and w not in sw})


st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.markdown("""
    <style>
    html, body, .stApp {background:#002244;color:#f1f1f1;}
    .stFileUploader, .stTextArea textarea {background:#1a2a44;color:#fff;border-radius:8px;}
    .stButton>button {background:#0d47a1;color:#fff;border:none;border-radius:8px;font-weight:600;}
    .highlight {color:#00ffff;font-weight:bold;}
    </style>
""", unsafe_allow_html=True)

st.title("AI Resume Analyzer")
uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "txt"], key="resume")

category_keywords_dict = {
    "Java Developer": ["java", "spring", "hibernate", "j2ee"],
    "Python Developer": ["python", "django", "flask", "pandas"],
    "Web Designing": ["html", "css", "javascript", "bootstrap"],
    "Data Science": ["machine", "learning", "data", "model", "prediction", "pandas", "scikit-learn"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "ci/cd"],
    "HR": ["recruitment", "interview", "onboarding", "hr"],
    "Testing": ["selenium", "automation", "testcase", "junit"],
}

if uploaded_file:
    resume_text = ""
    if uploaded_file.type == "application/pdf":
        resume_text = "".join(p.extract_text() or "" for p in PdfReader(uploaded_file).pages)
    else:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

    cleaned = clean_resume(resume_text)
    pred_id = clf.predict(tfid.transform([cleaned]))[0]
    category = le.inverse_transform([pred_id])[0]

    st.markdown(f"### 🎯 Predicted Category: **{category}**")
    found = highlight_keywords(cleaned, category_keywords_dict.get(category, []))

    if found:
        st.markdown("**🔍 Matched Keywords:** " + ", ".join(f"`{w}`" for w in found))
    else:
        st.markdown("⚠️ No category-specific keywords found.")

    st.download_button("⬇️ Download Category", category, file_name="category.txt", mime="text/plain")
    st.download_button("⬇️ Download Keywords", "\n".join(found) if found else "No keywords found.", file_name="keywords.txt", mime="text/plain")
