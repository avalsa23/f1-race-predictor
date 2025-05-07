import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🎯 Title
st.title("🏁 Formula 1 Race Position Predictor")

# 📄 Load dataset
df = pd.read_csv("f1_clean_dataset.csv")

# 🧹 Prepare features and target
X = df.drop(columns=[], errors="ignore")  # or just: X = df.copy()


y = df["position"]

# ✂️ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 Train model live
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 🎯 Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.markdown(f"### ✅ Model Accuracy: **{round(acc*100, 2)}%**")

# 🚦 Driver Prediction
st.markdown("### 🧠 Predict Race Finish Position")

driver_input = {}
for col in X.columns:
    dtype = df[col].dtype
    if dtype == "object":
        driver_input[col] = st.selectbox(f"{col}", options=sorted(df[col].unique()))
    else:
        driver_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

# 🔮 Make prediction
if st.button("Predict Finish Position"):
    input_df = pd.DataFrame([driver_input])
    result = model.predict(input_df)[0]
    st.success(f"🏆 Predicted Position: **{int(result)}**")

