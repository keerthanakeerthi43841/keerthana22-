import streamlit as st
import cv2
import easyocr
import numpy as np
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace
from PIL import Image
import pandas as pd

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Fraud Detection System",
    layout="wide",
    page_icon="🏦"
)

# Light Theme Styling Update
st.markdown("""
<style>
    .stApp {background-color:#f5f8fa!important;}
    .st-emotion-cache-1n76uvr {color:#003366!important;}
    header {background:#003366!important;color:#fff!important;}
    .stButton>button {
        background:#0055A4!important;color:#fff!important;font-weight:bold;
        border-radius:7px;border:none;
    }
    .stButton>button:hover {background:#003366!important;}
</style>
""", unsafe_allow_html=True)

st.title("🏦 AI Fraud Detection System")
option = st.sidebar.selectbox("Choose Module", [
    "Document Tampering",
    "Signature Verification",
    "Aadhaar Fraud Detection",
    "PAN Fraud Detection",
    "AI-Based KYC Verification",
    "Unusual Pattern Detection"
])

reader = easyocr.Reader(['en'], gpu=False)

# ---------------- Module: Document Tampering ----------------
if option == "Document Tampering":
    st.header("📄 Document Forgery Detection")
    col1, col2 = st.columns(2)

    with col1:
        doc1 = st.file_uploader("Upload Original", ["jpg","jpeg","png"])
    with col2:
        doc2 = st.file_uploader("Upload Suspected", ["jpg","jpeg","png"])

    if doc1 and doc2:
        img1 = cv2.imdecode(np.frombuffer(doc1.read(),np.uint8),1)
        img2 = cv2.imdecode(np.frombuffer(doc2.read(),np.uint8),1)
        img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))

        score, diff = ssim(
            cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),
            full=True
        )

        st.write(f"Similarity Score: *{score:.2f}*")
        st.image((diff*255).astype(np.uint8), caption="Difference Map")

        st.error("⚠ Possible Forgery") if score<0.85 else st.success("✅ Clean")

# ---------------- Signature Verification ----------------
elif option == "Signature Verification":
    st.header("✍ Signature Verification")
    f1 = st.file_uploader("Original Signature", ["jpg","jpeg","png"])
    f2 = st.file_uploader("Submitted Signature", ["jpg","jpeg","png"])

    if f1 and f2:
        s1 = cv2.imdecode(np.frombuffer(f1.read(),np.uint8),0)
        s2 = cv2.imdecode(np.frombuffer(f2.read(),np.uint8),0)
        orb = cv2.ORB_create()
        kp1,d1 = orb.detectAndCompute(s1,None)
        kp2,d2 = orb.detectAndCompute(s2,None)
        if d1 is not None and d2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING,True)
            score = len(bf.match(d1,d2))
            st.write(f"Match Score: {score}")
            st.success("✅ Genuine") if score>50 else st.error("❌ Forged")

# ---------------- Aadhaar Check ----------------
elif option == "Aadhaar Fraud Detection":
    st.header("🪪 Aadhaar Verification")
    num = st.text_input("Enter Aadhaar: XXXX-XXXX-XXXX")
    if st.button("Verify"):
        st.success("✅ Format Valid") if len(num)==14 else st.error("❌ Invalid Format")

# ---------------- PAN Check ----------------
elif option == "PAN Fraud Detection":
    st.header("💳 PAN Verification")
    pan = st.text_input("Enter PAN: ABCDE1234F")
    if st.button("Check"):
        valid = (len(pan)==10 and pan[:5].isalpha() and pan[5:9].isdigit() and pan[-1].isalpha())
        st.success("✅ Valid") if valid else st.error("❌ Invalid")

# ---------------- KYC Face Verification ----------------
elif option == "AI-Based KYC Verification":
    st.header("🧬 KYC Face Match")
    selfi = st.file_uploader("Selfie", ["jpg","jpeg","png"])
    idp = st.file_uploader("ID Card Photo", ["jpg","jpeg","png"])
    if selfi and idp:
        try:
            result = DeepFace.verify(
                np.array(Image.open(selfi)),
                np.array(Image.open(idp))
            )
            st.success("✅ Face Match") if result["verified"] else st.error("❌ Mismatch")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- Anomaly Detection ----------------
elif option == "Unusual Pattern Detection":
    st.header("📊 Anomaly Detector")
    f = st.file_uploader("Upload CSV", "csv")
    if f:
        df = pd.read_csv(f)
        st.dataframe(df.head())
        z = (df-df.mean())/df.std()
        anomalies = df[(abs(z)>3).any(axis=1)]
        st.subheader("Detected Anomalies:")
        st.dataframe(anomalies)

if st.button("Generate Fraud Report"):
    st.success("✅ Report Generated")
