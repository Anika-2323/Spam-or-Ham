import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso

st.title("Simple Lasso Spam Analysis")

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Total features
total_features = X.shape[1]
st.write("Total features:", total_features)

# Train Lasso
model = Lasso(alpha=0.1, max_iter=10000)
model.fit(X, y)

coef = model.coef_

non_zero = (coef != 0).sum()
zero = (coef == 0).sum()

st.write("Selected features:", non_zero)
st.write("Eliminated features:", zero)

# Alpha comparison
st.subheader("Alpha Comparison")

for alpha in [0.01, 0.1, 1]:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X, y)
    
    coef = model.coef_
    selected = (coef != 0).sum()
    
    st.write("Alpha:", alpha, "| Selected:", selected)

# Reduction %
reduction = ((total_features - non_zero) / total_features) * 100
st.write("Feature reduction: {:.2f}%".format(reduction))
