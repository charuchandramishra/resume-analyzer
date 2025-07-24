import streamlit as st
import pickle, re, os, ssl, smtplib
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# Ensure nltk data
nltk_path = os.path.expanduser("~/.nltk_data")
os.makedirs(nltk_path, exist_ok=True)
if nltk_path not in nltk.data.path:
    nltk.data.path.append(nltk_path)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass
for pkg in ("stopwords",):
    try:
        from nltk import data
        data.find(f"corpora/{pkg}")
    except LookupError:
        import nltk
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

def highlight_keywords(text, key_list):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    sw = set(stopwords.words("english"))
    return list({word for word in tokens if word in key_list and word not in sw})

def extract_info(text):
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.findall(r"\+?\d[\d\s\-]{8,15}", text)
    linkedin = re.findall(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+", text)
    github = re.findall(r"https?://(www\.)?github\.com/[A-Za-z0-9\-_]+", text)
    return {
        "Email": email[0] if email else "Not found",
        "Phone": phone[0] if phone else "Not found",
        "LinkedIn": linkedin[0] if linkedin else "Not found",
        "GitHub": github[0] if github else "Not found"
    }

def summarize_text(text, limit=60):
    sentences = re.split(r'[.!?]', text)
    summary = '. '.join([s.strip() for s in sentences if len(s.strip()) > 20][:3])
    return summary[:limit*3] + "..." if len(summary) > limit*3 else summary

def send_email(recipient, subject, body):
    sender = "your_email@example.com"
    password = "your_password"

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        return False

# Streamlit UI setup
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: #ffffff;
}
.stFileUploader, .stTextArea textarea {
    background-color: #142850;
    color: #ffffff;
    border-radius: 8px;
}
.stButton>button {
    background-color: #0d47a1;
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
}
.highlight { color: #00ffff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Resume Analyzer")

uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "txt"])

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
    if uploaded_file.type == "application/pdf":
        resume_text = "".join(p.extract_text() or "" for p in PdfReader(uploaded_file).pages)
    else:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

    cleaned = clean_resume(resume_text)
    pred_id = clf.predict(tfid.transform([cleaned]))[0]
    category = le.inverse_transform([pred_id])[0]

    st.subheader("🎯 Predicted Job Category:")
