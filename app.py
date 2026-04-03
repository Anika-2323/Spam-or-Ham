import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso

# ================================
# Title
# ================================
st.title("📩 SMS Spam Detection using Lasso")
st.write("Enter a message and check whether it's Spam or Ham")

# ================================
# Load Dataset
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# ================================
# TF-IDF Vectorization
# ================================
@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['message'])
    y = data['label']
    
    model = Lasso(alpha=0.1, max_iter=10000)
    model.fit(X, y)
    
    return vectorizer, model, X

vectorizer, model, X = train_model(df)

# ================================
# Show Feature Info
# ================================
st.subheader("📊 Feature Information")

total_features = X.shape[1]
coefficients = model.coef_

non_zero = np.sum(coefficients != 0)
zero = np.sum(coefficients == 0)
reduction = ((total_features - non_zero) / total_features) * 100

st.write("Total Features (TF-IDF):", total_features)
st.write("Selected Features (Non-zero):", non_zero)
st.write("Eliminated Features:", zero)
st.write("Feature Reduction (%): {:.2f}%".format(reduction))

# ================================
# User Input
# ================================
st.subheader("✍️ Enter your SMS")

user_input = st.text_area("Type your message here...")

# ================================
# Prediction
# ================================
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        
        # Convert regression output to classification
        if prediction >= 0.5:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is HAM (Not Spam)")

# ================================
# Footer
# ================================
st.write("---")
st.caption("Built with ❤️ using Streamlit + Lasso Regression")
