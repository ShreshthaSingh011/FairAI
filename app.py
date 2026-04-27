import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
# ---- CONFIG ----
st.set_page_config(page_title="FairAI", page_icon="⚖️", layout="wide")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---- SESSION STATE ----
if "bias" not in st.session_state:
    st.session_state.bias = None
if "new_bias" not in st.session_state:
    st.session_state.new_bias = None

# ---- HEADER ----
st.title("⚖️ FairAI")
st.caption("AI-powered fairness auditing system with detection, explanation, and mitigation")

st.markdown("---")

# ---- SIDEBAR INPUT ----
with st.sidebar:
    st.header("📂 Input Panel")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Attribute", df.columns)

        run_analysis = st.button("🚀 Run Analysis")
        explain = st.button("🤖 Explain Bias")
        mitigate = st.button("🛠️ Apply Mitigation")

# ---- MAIN DASHBOARD ----
if file:

    col1, col2 = st.columns([1, 2])

    # ---- LEFT: DATA PREVIEW ----
    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

    # ---- RIGHT: RESULTS ----
    with col2:

        # ---- RUN ANALYSIS ----
        if run_analysis:
            try:
                X = df.drop(columns=[target])
                y = df[target]

                X = pd.get_dummies(X, drop_first=True)

                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                preds = model.predict(X)

                bias = demographic_parity_difference(
                    y_true=y,
                    y_pred=preds,
                    sensitive_features=df[sensitive]
                )

                st.session_state.bias = bias

            except Exception as e:
                st.error(f"Error: {e}")

        # ---- SHOW RESULTS ----
        if st.session_state.bias is not None:

            st.subheader("📈 Bias Analysis")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Bias Score", round(st.session_state.bias, 2))

            # Determine level
            if st.session_state.bias < 0.1:
                level = "Low"
            elif st.session_state.bias < 0.3:
                level = "Moderate"
            else:
                level = "High"

            with colB:
                st.metric("Bias Level", level)

            if level == "High":
                st.error("⚠️ High Bias Detected")
            elif level == "Moderate":
                st.warning("⚠️ Moderate Bias")
            else:
                st.success("✅ Low Bias")

            st.write("This indicates disparity in predictions across groups.")

            st.markdown("---")

        # ---- GEMINI EXPLANATION ----
        if explain:

            if st.session_state.bias is not None:

                st.subheader("🤖 AI Explanation")

                try:
                    prompt = f"""
                    The bias score is {st.session_state.bias}.
                    Explain what this means, why it occurs, and how to reduce it.
                    """

                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt
                    )

                    st.write(response.text)

                except Exception:
                    st.info("Using built-in AI explanation (fallback mode).")

                    st.write(f"""
                    The model shows a bias score of **{round(st.session_state.bias,2)}**, 
                    indicating disparity between groups.

                    **Possible causes:**
                    - Imbalanced training data  
                    - Correlation with sensitive features  

                    **Fixes:**
                    - Balance dataset  
                    - Remove biased features  
                    - Use fairness-aware algorithms  
                    """)

        # ---- MITIGATION ----
        if mitigate:

            try:
                df_balanced = df.groupby([target, sensitive]).apply(
                    lambda x: x.sample(
                        df.groupby([target, sensitive]).size().min(),
                        replace=True
                    )
                ).reset_index(drop=True)

                X_bal = df_balanced.drop(columns=[target])
                y_bal = df_balanced[target]

                X_bal = pd.get_dummies(X_bal, drop_first=True)

                model = LogisticRegression(max_iter=1000)
                model.fit(X_bal, y_bal)
                preds_bal = model.predict(X_bal)

                new_bias = demographic_parity_difference(
                    y_true=y_bal,
                    y_pred=preds_bal,
                    sensitive_features=df_balanced[sensitive]
                )

                st.session_state.new_bias = new_bias

            except Exception as e:
                st.error(f"Mitigation Error: {e}")

        # ---- SHOW MITIGATION ----
        if st.session_state.new_bias is not None:

            st.subheader("🛠️ After Mitigation")

            st.metric("New Bias Score", round(st.session_state.new_bias, 2))

            st.success(
                f"Bias reduced from {round(st.session_state.bias,2)} → {round(st.session_state.new_bias,2)}"
            )

            # ---- CHART ----
            fig, ax = plt.subplots()

            values = [st.session_state.bias, st.session_state.new_bias]
            labels = ["Before", "After"]

            bars = ax.bar(labels, values)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02,
                        round(yval,2), ha='center')

            ax.set_ylim(0, 1)
            ax.set_ylabel("Bias Score")
            ax.set_title("Bias Reduction")
            ax.spines[['top','right']].set_visible(False)

            st.pyplot(fig)

# ---- FOOTER ----
st.markdown("---")
st.caption("Powered by Google AI with built-in fallback for reliability")