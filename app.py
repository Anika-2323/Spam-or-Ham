# ================================
# 1. Import Libraries
# ================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Dataset Loaded Successfully!\n")

# ================================
# 3. TF-IDF Vectorization
# ================================
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['message'])
y = df['label']

# Total features
total_features = X.shape[1]
print("Total number of features created by TF-IDF:", total_features)
print()

# ================================
# 4. Lasso Model (alpha = 0.1)
# ================================
lasso_01 = Lasso(alpha=0.1)
lasso_01.fit(X, y)

coef_01 = lasso_01.coef_

non_zero_01 = (coef_01 != 0).sum()
zero_01 = (coef_01 == 0).sum()

print("Results for alpha = 0.1")
print("Non-zero coefficients (selected features):", non_zero_01)
print("Zero coefficients (eliminated features):", zero_01)
print()

# ================================
# 5. Different Alpha Values
# ================================
alphas = [0.01, 0.1, 1]

results = []

for alpha in alphas:
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    
    coef = model.coef_
    
    non_zero = (coef != 0).sum()
    zero = (coef == 0).sum()
    
    reduction = ((total_features - non_zero) / total_features) * 100
    
    results.append((alpha, non_zero, zero, reduction))

# ================================
# 6. Display Results
# ================================
print("Comparison of Lasso with different alpha values:\n")

for res in results:
    print("Alpha =", res[0])
    print("Selected features (non-zero):", res[1])
    print("Eliminated features:", res[2])
    print("Percentage reduction: {:.2f}%".format(res[3]))
    print("-----------------------------------")

# ================================
# 7. Final Percentage Reduction (alpha=0.1)
# ================================
reduction_01 = ((total_features - non_zero_01) / total_features) * 100

print("\nFinal Percentage Reduction for alpha=0.1: {:.2f}%".format(reduction_01))
