import streamlit as st
import pickle
import re
import nltk
from io import StringIO
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tfid = pickle.load(open('tfid.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

def clean_resume(resume_text):
    text = re.sub(r'https\S+', ' ', resume_text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+\s', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub('[%s]' % re.escape("""!#$&'()+,<>?;:=@[]^_`{|}~"""), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def highlight_keywords(resume_text, category_keywords):
    tokens = word_tokenize(resume_text.lower())
    keywords = [word for word in tokens if word in category_keywords and word not in stopwords.words('english')]
    return list(set(keywords))

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #002244 !important;
        color: #f1f1f1 !important;
    }

    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }

    .block-container {
        padding: 2rem 1rem 1rem;
        background-color: #002244 !important;
    }

    .main {
        background-color: #002244 !important;
        padding: 2rem;
        border-radius: 10px;
    }

    .stFileUploader, .stTextInput, .stTextArea, .stSelectbox, .stMultiselect, .stNumberInput {
        background-color: #1a2a44 !important;
        color: #ffffff !important;
        border-radius: 8px;
    }

    .stTextArea textarea {
        background-color: #1a2a44 !important;
        color: #ffffff !important;
    }

    .stButton>button {
        background-color: #0d47a1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }

    .stAlert {
        background-color: #003366;
        color: #ffffff;
    }

    .highlight {
        color: #00ffff;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🧠 AI Resume Analyzer")
uploaded_file = st.file_uploader('📄 Upload Resume', type=['txt', 'pdf'])

category_keywords_dict = {
    "Java Developer": ['java', 'spring', 'hibernate', 'j2ee'],
    "Python Developer": ['python', 'django', 'flask', 'pandas'],
    "Web Designing": ['html', 'css', 'javascript', 'bootstrap'],
    "Data Science": ['machine', 'learning', 'data', 'model', 'prediction', 'pandas', 'scikit-learn'],
    "DevOps Engineer": ['docker', 'kubernetes', 'jenkins', 'ci/cd'],
    "HR": ['recruitment', 'interview', 'onboarding', 'hr'],
    "Testing": ['selenium', 'automation', 'testcase', 'junit'],
}

if uploaded_file is not None:
    if uploaded_file.type == 'application/pdf':
        reader = PdfReader(uploaded_file)
        resume_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode('utf-8', errors='ignore')

    cleaned_resume = clean_resume(resume_text)
    input_feature = tfid.transform([cleaned_resume])
    prediction_id = clf.predict(input_feature)[0]
    category_name = le.inverse_transform([prediction_id])[0]

    st.markdown(f"<h3 style='color:#90caf9;'>Predicted Category: {category_name}</h3>", unsafe_allow_html=True)

    keywords = highlight_keywords(cleaned_resume, category_keywords_dict.get(category_name, []))
    if keywords:
        st.markdown("**🔍 Matched Keywords in Resume:**")
        st.markdown(", ".join([f"`{kw}`" for kw in keywords]))
    else:
        st.markdown("⚠️ No category-specific keywords found in the resume.")
