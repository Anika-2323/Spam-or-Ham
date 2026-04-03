import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Total features
total_features = X.shape[1]
print("Total features:", total_features)

# Train Lasso
model = Lasso(alpha=0.1, max_iter=10000)
model.fit(X, y)

# Coefficients
coef = model.coef_

# Count features
non_zero = (coef != 0).sum()
zero = (coef == 0).sum()

print("Selected features:", non_zero)
print("Eliminated features:", zero)

# Try different alpha values
for alpha in [0.01, 0.1, 1]:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X, y)
    
    coef = model.coef_
    selected = (coef != 0).sum()
    
    print("Alpha:", alpha, "| Selected features:", selected)

# Percentage reduction (for alpha=0.1)
reduction = ((total_features - non_zero) / total_features) * 100
print("Feature reduction: {:.2f}%".format(reduction))
