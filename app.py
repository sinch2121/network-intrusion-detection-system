import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Network Intrusion Detection System", layout="wide")

st.title("🔐 Network Intrusion Detection System")
st.write("Machine Learning-based Cybersecurity Attack Detection")

# Load model
model = joblib.load("model.pkl")

# Sidebar
st.sidebar.header("Upload Network Data CSV")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    predictions = model.predict(data)
    data["Prediction"] = predictions
    data["Prediction"] = data["Prediction"].map({0: "Normal", 1: "Attack"})

    st.subheader("Prediction Results")
    st.dataframe(data.head())

    attack_count = (data["Prediction"] == "Attack").sum()
    normal_count = (data["Prediction"] == "Normal").sum()

    st.success(f"Normal Traffic: {normal_count}")
    st.error(f"Attack Traffic: {attack_count}")

    # Download results
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results",
        csv,
        "predictions.csv",
        "text/csv",
        key="download-csv"
    )

# Divider
st.markdown("---")
st.header("📊 Model Evaluation")

if st.button("Show Confusion Matrix & Feature Importance"):

    st.write("Loading evaluation data...")

    columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
    ]

    test = pd.read_csv("data/KDDTest+.txt", names=columns).sample(1000, random_state=42)
    test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    X_test = test.drop(['label','difficulty'], axis=1)
    y_test = test['label']

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Importance")

    # Extract classifier from pipeline
    rf = model.named_steps["classifier"]
    importances = rf.feature_importances_

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    fig2, ax2 = plt.subplots()
    ax2.barh(importance_df["Feature"], importance_df["Importance"])
    ax2.invert_yaxis()
    st.pyplot(fig2)