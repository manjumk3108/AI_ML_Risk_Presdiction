import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 0. Basic Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Project Risk Prediction",
    layout="wide"
)

st.title("AI-Based Project Risk Prediction for Project Management")

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to section:",
    [
        "Introduction",
        "Project Overview",
        "Data & EDA",
        "Model Performance",
        "Feature Importance",
        "Demo",
        "Results & Discussion",
        "Conclusion & Recommendations",
    ],
)

# ---------------------------------------------------------
# 1. Data Loading, Cleaning, and Model Preparation
# ---------------------------------------------------------
@st.cache_data
def load_data(path: str = "data/raw_data/project_risk_raw_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleaning (same as notebook)
    if "Tech_Environment_Stability" in df.columns:
        df = df.drop(columns=["Tech_Environment_Stability"])

    for col in ["Change_Control_Maturity", "Risk_Management_Maturity"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    # -------------------------------------------------
    # Define Input Features (X) and Target Variable (y)
    # -------------------------------------------------
    df_model = df.drop(columns=["Project_ID"])  # Project ID will not be useful

    X = df_model.drop(columns=["Risk_Level"])
    y = df_model["Risk_Level"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocessing – scale numeric features and one-hot encode categorical features

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 1) Logistic Regression
    # -----------------------------
    log_reg_pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    log_reg_pipe.fit(X_train, y_train)
    y_pred_lr = log_reg_pipe.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=False)
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=["Critical", "High", "Medium", "Low"])

    # -------------------------
    # 2) Random Forest 
    # -------------------------
    rf_pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=False)
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=["Critical", "High", "Medium", "Low"])

    # ----------------------------------------------------
    # 3) Hierarchical Random Forest
    #    Stage 1: high-risk? (High/Critical vs Medium/Low)
    #    Stage 2a: Critical vs High
    #    Stage 2b: Medium vs Low
    # ----------------------------------------------------
    # Stage 1
    y_highflag = y_train.isin(["High", "Critical"]).astype(int)
    rf_stage1 = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42
                ),
            ),
        ]
    )
    rf_stage1.fit(X_train, y_highflag)

    # Stage 2 for high-branch (High vs Critical)
    mask_high = y_train.isin(["High", "Critical"])
    X_train_high = X_train[mask_high]
    y_train_high = y_train[mask_high].map({"High": 0, "Critical": 1})

    rf_stage2_high = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42
                ),
            ),
        ]
    )
    rf_stage2_high.fit(X_train_high, y_train_high)

    # Stage 2 for low-branch (Low vs Medium)
    mask_low = y_train.isin(["Low", "Medium"])
    X_train_low = X_train[mask_low]
    y_train_low = y_train[mask_low].map({"Low": 0, "Medium": 1})

    rf_stage2_low = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42
                ),
            ),
        ]
    )
    rf_stage2_low.fit(X_train_low, y_train_low)

    # Predict with hierarchical model
    def hier_predict(X_rows: pd.DataFrame) -> np.ndarray:
        preds = []
        for i in range(len(X_rows)):
            x_row = X_rows.iloc[i : i + 1]

            stage1 = rf_stage1.predict(x_row)[0]
            if stage1 == 1:
                # high-risk branch
                lab = rf_stage2_high.predict(x_row)[0]
                preds.append("High" if lab == 0 else "Critical")
            else:
                lab = rf_stage2_low.predict(x_row)[0]
                preds.append("Low" if lab == 0 else "Medium")
        return np.array(preds)

    y_pred_hier = hier_predict(X_test)
    acc_hier = accuracy_score(y_test, y_pred_hier)
    report_hier = classification_report(y_test, y_pred_hier, output_dict=False)
    cm_hier = confusion_matrix(y_test, y_pred_hier, labels=["Critical", "High", "Medium", "Low"])

    metrics = {
        "lr": {"acc": acc_lr, "report": report_lr, "cm": cm_lr},
        "rf": {"acc": acc_rf, "report": report_rf, "cm": cm_rf},
        "hier": {"acc": acc_hier, "report": report_hier, "cm": cm_hier},
    }

    model_objects = {
        "preprocessor": preprocessor,
        "log_reg": log_reg_pipe,
        "rf_flat": rf_pipe,
        "rf_stage1": rf_stage1,
        "rf_stage2_high": rf_stage2_high,
        "rf_stage2_low": rf_stage2_low,
    }

    return X, y, X_train, X_test, y_train, y_test, metrics, model_objects, numeric_cols, categorical_cols


df = load_data()
X, y, X_train, X_test, y_train, y_test, metrics, models, numeric_cols, categorical_cols = train_models(
    df
)

# ---------------------------------------------------------
# Supporting Function for confusion matrix as heatmap
# ---------------------------------------------------------
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def hierarchical_predict_single(row: pd.Series) -> str:
    """Use the trained hierarchical RF to predict 1 project."""
    x_df = row.to_frame().T  # keep as DataFrame
    stage1 = models["rf_stage1"].predict(x_df)[0]

    if stage1 == 1:
        lab = models["rf_stage2_high"].predict(x_df)[0]
        return "High" if lab == 0 else "Critical"
    else:
        lab = models["rf_stage2_low"].predict(x_df)[0]
        return "Low" if lab == 0 else "Medium"

# ---------------------------------------------------------
# SECTION: Introduction
# ---------------------------------------------------------
if section == "Introduction":
    st.header("Introduction")

    # --- Course & General Info ---
    with st.container():
        st.subheader("Course & Project Information")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            **Name:** Manjunath Mallikarjun Kendhuli  
            **Matriculation Number:** 400890897  
            """)

        with col_b:
            st.markdown("""
            **Course:** Artificial Intelligence & Machine Learning  
            **Professor:** Prof. Dr. Aleena Baby  
            """)



# ---------------------------------------------------------
# SECTION: PROJECT OVERVIEW
# ---------------------------------------------------------
elif section == "Project Overview":
    st.header("Project Overview")

    # --- USING TABS FOR CLEAN LAYOUT ---
    tab1, tab2 = st.tabs(["Business Problem", "Goal of this Model"])

    with tab1:
        st.subheader("Business Problem")
        st.markdown("""
        Modern organizations run **multiple projects in parallel** – IT, construction, R&D, operations.

        Managing these projects is challenging because:

        - Each project has **different teams, budgets, timelines, and dependencies**.
        - Many factors can lead to failure:
          - Scope changes  
          - Schedule pressure  
          - Resource constraints  
          - Regulatory / compliance risk  
        - It is **hard for a Project Manager or PMO** to continuously monitor all projects
          and identify which ones are becoming **High or Critical risk**.

        As a result:
        - Some risks are detected **too late**.
        - Budget overruns, delays, or project cancellations can happen.
        """)

    with tab2:
        st.subheader("Goal of this Model")
        st.markdown("""
        The goal of this project is to build an **AI-based Project Risk Prediction System** that can:

        - Predict the overall **Risk Level** of a project:
          - Critical  
          - High  
          - Medium  
          - Low  

        and help Project Managers / PMOs to:

        - Prioritise which projects need **urgent attention**  
        - Plan **mitigation actions** early (e.g., add resources, reduce scope)  
        - Monitor **key risk drivers** such as:
          - Project complexity  
          - Schedule pressure  
          - Team experience  
          - Organizational change frequency  
        """)

# ---------------------------------------------------------
# SECTION: DATA & EDA
# ---------------------------------------------------------
elif section == "Data & EDA":
    st.header("Data & Exploratory Analysis")

    # --- Dataset Snapshot ---
    st.subheader("Dataset Snapshot")
    st.write(f"- Number of projects: **{df.shape[0]}**")
    st.write(f"- Number of columns: **{df.shape[1]}**")
    st.dataframe(df.head())

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Numerical Feature Distribution", "Outlier Detection", "Categorical Feature Distribution", "Feature vs Risk Level"]
    )

    # ---------- Tab 1: Numeric distributions ----------
    with tab1:
        st.subheader("Numerical Feature Distribution (Histogram)")

        num_col = st.selectbox(
            "Choose a numeric column",
            options=numeric_cols,
            index=0,
            key="num_dist_col",
        )

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df[num_col], bins=20, kde=True, alpha=0.8)
        ax.set_title(f"Distribution of {num_col}", fontsize=12)
        st.pyplot(fig)

    # ---------- Tab 2: Outlier plots ----------
    with tab2:
        st.subheader("Outlier Detection (Boxplot)")

        out_col = st.selectbox(
            "Choose a numeric column",
            options=numeric_cols,
            index=0,
            key="outlier_col",
        )

        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.boxplot(x=df[out_col], vert=False)
        ax.set_title(f"Outliers in {out_col}", fontsize=12)
        st.pyplot(fig)

    # ---------- Tab 3: Categorical distributions ----------
    with tab3:
        st.subheader("Categorical Feature Distribution")

        cat_col = st.selectbox(
            "Choose a categorical column",
            options=categorical_cols + ["Risk_Level"],
            index=len(categorical_cols),  # default = Risk_Level
            key="cat_dist_col",
        )

        fig, ax = plt.subplots(figsize=(6, 3))
        order = df[cat_col].value_counts().index
        sns.countplot(y=df[cat_col], order=order, ax=ax)
        ax.set_title(f"Distribution of {cat_col}", fontsize=12)
        st.pyplot(fig)
    
    st.markdown("---")

    st.subheader("EDA – Key Takeaways")
    st.markdown("""
    - Data contains both **numerical and categorical** project features.
    - **Outliers exist** in budget, team size, and timeline.
    - **Risk levels are slightly imbalanced**, with Medium & High dominating.
    - **Agile and Scrum are the most common methodologies.**
    - EDA helped identify **key risk-driving factors** and guided model selection.
    """)

    # ---------------Tab 4: Feature vs Risk_Level-----------------------

    with tab4:
        st.subheader("How a Feature Relates to Risk_Level")

        feature_cols = [c for c in df.columns if c != "Risk_Level"]

        feat_choice = st.selectbox("Choose a feature", feature_cols)

        # Risk level order (if present)
        risk_order = ["Low", "Medium", "High", "Critical"]
        present_levels = [r for r in risk_order if r in df["Risk_Level"].unique()]

        # Numeric feature → boxplot by risk level
        if pd.api.types.is_numeric_dtype(df[feat_choice]):
            st.write(f"Boxplot of **{feat_choice}** grouped by **Risk_Level**")

            grouped = [df.loc[df["Risk_Level"] == r, feat_choice]
                       for r in present_levels]

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.boxplot(grouped, labels=present_levels)
            ax.set_xlabel("Risk_Level")
            ax.set_ylabel(feat_choice)
            ax.set_title(f"{feat_choice} distribution by Risk_Level", fontsize=12)
            st.pyplot(fig, use_container_width=True)

        # Categorical feature → stacked bar (percentage)
        else:
            st.write(
                f"Stacked bar showing percentage of **Risk_Level** "
                f"within each category of **{feat_choice}**"
            )

            ct = pd.crosstab(df[feat_choice], df["Risk_Level"], normalize="index") * 100
            # keep top 8 categories for readability
            ct = ct.sort_values(by=ct.columns.tolist(), ascending=False).head(8)

            fig, ax = plt.subplots(figsize=(6, 3))
            bottom = np.zeros(len(ct))
            for r in present_levels:
                if r in ct.columns:
                    ax.bar(ct.index, ct[r], bottom=bottom, label=r)
                    bottom += ct[r].values

            ax.set_ylabel("Percentage within category (%)")
            ax.set_title(f"{feat_choice} vs Risk_Level", fontsize=12)
            ax.set_xticklabels(ct.index, rotation=45, ha="right")
            ax.legend(title="Risk_Level", bbox_to_anchor=(1.02, 1), loc="upper left")
            st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------
# SECTION: MODEL PERFORMANCE
# ---------------------------------------------------------
elif section == "Model Performance":
    st.header("Model Performance")

    submodel = st.radio(
        "Select model:",
        ["Logistic Regression", "Random Forest", "Hierarchical Random Forest"],
    )

    if submodel == "Logistic Regression":
        st.subheader("1. Logistic Regression")
        st.write(f"**Accuracy on test set:** {metrics['lr']['acc']:.3f}")

        st.text("Classification report:")
        st.text(metrics["lr"]["report"])

        st.text("Confusion Matrix:")
        plot_confusion_matrix(metrics["lr"]["cm"], ["Critical", "High", "Medium", "Low"])



    elif submodel == "Random Forest":
        st.subheader("2. Random Forest")
        st.write(f"**Accuracy on test set:** {metrics['rf']['acc']:.3f}")

        st.text("Classification report:")
        st.text(metrics["rf"]["report"])

        st.text("Confusion Matrix:")
        plot_confusion_matrix(metrics["rf"]["cm"], ["Critical", "High", "Medium", "Low"])



    else:
        st.subheader("3. Hierarchical Random Forest")
        st.write(f"**Accuracy on test set:** {metrics['hier']['acc']:.3f}")

        st.text("Classification report:")
        st.text(metrics["hier"]["report"])

        st.text("Confusion Matrix:")
        plot_confusion_matrix(metrics["hier"]["cm"], ["Critical", "High", "Medium", "Low"])



# ---------------------------------------------------------
# SECTION: FEATURE IMPORTANCE
# ---------------------------------------------------------
elif section == "Feature Importance":
    st.header("Feature Importance")

    # Get feature importance from Random Forest model
    rf_model = models["rf_flat"].named_steps["clf"]
    preprocessor = models["rf_flat"].named_steps["prep"]

    # Get expanded feature names
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names

    importances = rf_model.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": all_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    st.write("Top 15 most important engineered features:")
    st.dataframe(feat_imp)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        y="feature",
        x="importance",
        data=feat_imp,
        ax=ax,
    )
    ax.set_title("Top 15 Feature Importances")
    st.pyplot(fig)

# ---------------------------------------------------------
# SECTION: DEMO – 5 INPUTS + PREDICTION
# ---------------------------------------------------------
elif section == "Demo":
    st.header("Interactive Demo: Predict Project Risk")

    st.markdown(
        """
        Choose **five key features** and let the three models
        predict the **Risk Level** for this hypothetical project.

        The remaining features are kept at their **typical (mean/mode) values**
        from the historical data.
        """
    )

    # Choose some intuitive features for the UI
    ui_features = {
        "Project_Type": df["Project_Type"].unique().tolist(),
        "Methodology_Used": df["Methodology_Used"].unique().tolist(),
        "Team_Experience_Level": df["Team_Experience_Level"].unique().tolist(),
        "Team_Size": (int(df["Team_Size"].min()), int(df["Team_Size"].max())),
        "Complexity_Score": (float(df["Complexity_Score"].min()), float(df["Complexity_Score"].max())),
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        proj_type = st.selectbox("Project_Type", ui_features["Project_Type"])
        methodology = st.selectbox("Methodology_Used", ui_features["Methodology_Used"])
    with col2:
        team_exp = st.selectbox("Team_Experience_Level", ui_features["Team_Experience_Level"])
        team_size = st.slider(
            "Team_Size",
            min_value=ui_features["Team_Size"][0],
            max_value=ui_features["Team_Size"][1],
            value=int(df["Team_Size"].median()),
        )
    with col3:
        complexity = st.slider(
            "Complexity_Score",
            min_value=round(ui_features["Complexity_Score"][0], 1),
            max_value=round(ui_features["Complexity_Score"][1], 1),
            value=float(df["Complexity_Score"].median()),
            step=0.1,
        )

    # Build a "base" row with typical values (mean/mode)
    base_row = {}
    for c in X.columns:
        if c in ["Project_Type", "Methodology_Used", "Team_Experience_Level", "Team_Size", "Complexity_Score"]:
            continue
        if c in numeric_cols:
            base_row[c] = df[c].mean()
        else:
            base_row[c] = df[c].mode()[0]

    # Override with user choices
    base_row["Project_Type"] = proj_type
    base_row["Methodology_Used"] = methodology
    base_row["Team_Experience_Level"] = team_exp
    base_row["Team_Size"] = team_size
    base_row["Complexity_Score"] = complexity

    input_df = pd.DataFrame([base_row])

    if st.button("Predict Risk Level"):
        # 1) Logistic Regression
        pred_lr = models["log_reg"].predict(input_df)[0]

        # 2) Random Forest (flat)
        pred_rf = models["rf_flat"].predict(input_df)[0]

        # 3) Hierarchical Random Forest
        pred_hier = hierarchical_predict_single(input_df.iloc[0])

        st.subheader("Predictions")
        st.write(f"**Logistic Regression:** {pred_lr}  (test accuracy: {metrics['lr']['acc']:.3f})")
        st.write(f"**Random Forest (flat):** {pred_rf}  (test accuracy: {metrics['rf']['acc']:.3f})")
        st.write(
            f"**Hierarchical Random Forest (2-Stage):** {pred_hier}  "
            f"(test accuracy: {metrics['hier']['acc']:.3f})"
        )


# ---------------------------------------------------------
# SECTION: RESULTS & DISCUSSIONS
# ---------------------------------------------------------
elif section == "Results & Discussion":
    st.header("Results & Discussion")

    st.subheader("Model Performance Summary")

    st.markdown("""
    We compared three machine learning models to predict project risk levels:

     - **Logistic Regression** → *Best performing model*
       - **Test Accuracy: 72.1%**

     - **Random Forest**:  
       - **Test Accuracy: 55.6%**

     - **Hierarchical Random Forest**  
       - **Test Accuracy: 54.9%**

     Final Model Selected for Business Use: **Logistic Regression**
    """)

    st.subheader("Key Findings")

    st.markdown("""
    - Logistic Regression performed **better than tree-based models** on this dataset.
    - The model predicts:
      - **Low & Medium risks more accurately**
      - **Critical & High risks reasonably well**
    - Project risk is strongly influenced by:
      - Complexity Score
      - Team Turnover rate
      - Previous Delivery Success Rate
      - Estimated Timeline
      - Project Budget
    """)

    st.subheader("Business Interpretation")

    st.markdown("""
    - The model provides early warning signals before a project moves into **High or Critical risk**.
    - This allows Project Managers and PMOs to:
      - Prioritize Critical and High-risk projects
      - Allocate senior and specialized resources early
      - Control cost overruns and schedule delays
    - The model supports data-driven decision making instead of relying only on experience or intuition.
    """)

    st.subheader("Limitations & Assumptions")

    st.markdown("""
    - Dataset is **synthetic / simulated**
    - Real-world data may contain:
      - Noise
      - Human reporting bias
    - The model assumes:
      - Past project patterns repeat in future projects
 """)


# ---------------------------------------------------------
# SECTION: CONCLUSION
# ---------------------------------------------------------
elif section == "Conclusion & Recommendations":
    st.header("Conclusion & Recommendations")

    st.subheader("Project Summary")

    st.markdown("""
    In this project, we successfully built an:
    **AI-based Project Risk Prediction System**

    The model:
    - Predicts project risk as:
      - **Critical | High | Medium | Low**
    - Achieved:
      -**72% accuracy using Logistic Regression**
    """)

    st.subheader("What Organizations Should Do")

    st.markdown("""
    - Integrate the model into:
     - Project Management Dashboards
    - Use predictions to:
      - Identify **high-risk projects early**
      - Improve monitoring & escalation
      - Optimize cost & resource allocation
    """)

    st.subheader("Future Work")

    st.markdown("""
    - Use **real company project data**
    - Add:
      - Live project tracking
      - Automated alerts
    - Deploy as:
      - Internal web system
      - Enterprise project risk engine
    """)
