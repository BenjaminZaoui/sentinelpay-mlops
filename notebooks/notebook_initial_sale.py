import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

# Chargement
data = pd.read_csv("data/creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Separation AVANT la mise a l'echelle, stratifiee, graine fixee
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Mise a l'echelle : fit sur le train uniquement
scaler = StandardScaler()
X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

# Entrainement
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# Evaluation : metriques adaptees a la fraude
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("Recall    :", recall_score(y_test, pred))
print("Precision :", precision_score(y_test, pred))
print("F1        :", f1_score(y_test, pred))
print("PR AUC    :", average_precision_score(y_test, proba))