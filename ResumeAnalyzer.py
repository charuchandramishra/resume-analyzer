import streamlit as st
import pickle, re, os, ssl, smtplib
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

nltk_path = os.path.expanduser("~/.nltk_data")
os.makedirs(nltk_path, exist_ok=True)
if nltk_path not in nltk.data.path:
    nltk.data.path.append(nltk_path)

try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_path)

tfid = pickle.load(open("tfid.pkl", "rb"))
clf = pickle.load(open("clf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def clean_resume(text):
    text = re.sub(r"https?\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"[!#$%&'()*+,<=>?@\[\]^_`{|}~]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def highlight_keywords(text, keywords):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    sw = set(stopwords.words("english"))
    return list({w for w in tokens if w in keywords and w not in sw})

def extract_contact_details(text):
    email = re.findall(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.findall(r"\b[0-9]{10}\b", text)
    linkedin = re.findall(r"https?://[\w\./-]*linkedin[\w\./-]*", text)
    github = re.findall(r"https?://[\w\./-]*github[\w\./-]*", text)
    return {
        "Email": email[0] if email else "Not Found",
        "Phone": phone[0] if phone else "Not Found",
        "LinkedIn": linkedin[0] if linkedin else "Not Found",
        "GitHub": github[0] if github else "Not Found"
    }

def summarize_text(text, max_sentences=5):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

def send_email(receiver, subject, body):
    sender = "your_email@gmail.com"
    password = "your_password"
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        return False

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.markdown("""
    <style>
    html, body, .stApp {
        background: linear-gradient(to right, #002244, #005792);
        color: #ffffff;
    }
    .stFileUploader, .stTextArea textarea {
        background:#0b2545;
        color:#ffffff;
        border-radius:8px;
    }
    .stButton>button {
        background:#1e88e5;
        color:#fff;
        border:none;
        border-radius:8px;
        font-weight:600;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI Resume Analyzer")
uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "txt"])

category_keywords_dict = {
    "Java Developer": ["java", "spring", "hibernate", "j2ee"],
    "Python Developer": ["python", "django", "flask", "pandas"],
    "Web Designing": ["html", "css", "javascript", "bootstrap"],
    "Data Science": ["machine", "learning", "data", "model", "prediction", "pandas", "scikit-learn"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "ci/cd"],
    "HR": ["recruitment", "interview", "onboarding", "hr"],
    "Testing": ["selenium", "automation", "testcase", "junit"]
}

if uploaded_file:
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

    contact_info = extract_contact_details(resume_text)
    st.markdown("### 📬 Contact Details")
    for k, v in contact_info.items():
        st.write(f"**{k}:** {v}")

    summary = summarize_text(cleaned)
    st.markdown("### 📝 Resume Summary")
    st.info(summary)

    #st.download_button("⬇️ Download Category", category, file_name="category.txt", mime="text/plain")
    #st.download_button("⬇️ Download Keywords", "\n".join(found) if found else "No keywords found.", file_name="keywords.txt", mime="text/plain")

    with st.expander("📧 Send Summary to Email"):
        email_input = st.text_input("Enter recipient email")
        if st.button("Send Email") and email_input:
            success = send_email(email_input, "Resume Summary", summary)
            if success:
                st.success("✅ Email sent successfully!")
            else:
                st.error("❌ Failed to send email.")
