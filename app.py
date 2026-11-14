import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="OptiCrop | Human-Centered AI Crop Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATH RESOLUTION & SELF-HEALING MODEL LOADER
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "03_Dataset", "Crop_recommendation.csv")
SCALER_PATH = os.path.join(BASE_DIR, "05_Preprocessing", "scaler.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "06_Model", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
EDA_PLOTS_DIR = os.path.join(BASE_DIR, "04_Data_Analysis", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "06_Model", "results")

SEASON_MAPPING = {
    "rice": "Kharif (Monsoon)",
    "maize": "Kharif (Monsoon)",
    "cotton": "Kharif (Monsoon)",
    "pigeonpeas": "Kharif (Monsoon)",
    "mungbean": "Kharif (Monsoon)",
    "blackgram": "Kharif (Monsoon)",
    "jute": "Kharif (Monsoon)",
    "mothbeans": "Kharif (Monsoon)",
    "chickpea": "Rabi (Winter)",
    "lentil": "Rabi (Winter)",
    "kidneybeans": "Rabi (Winter)",
    "coffee": "Perennial",
    "coconut": "Perennial",
    "banana": "Perennial",
    "orange": "Perennial",
    "apple": "Perennial",
    "mango": "Perennial",
    "grapes": "Perennial",
    "papaya": "Perennial",
    "watermelon": "Summer",
    "muskmelon": "Summer",
    "pomegranate": "Summer"
}


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df["Season"] = df["label"].map(lambda x: SEASON_MAPPING.get(x, "Perennial"))
    return df


@st.cache_resource
def load_or_train_artifacts():
    df = load_dataset()
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LE_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            le = joblib.load(LE_PATH)
            return model, scaler, le, df
        except Exception:
            pass

    df_clean = df.copy()
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df_clean[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    X = df_clean[numeric_cols]
    y = df_clean["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LE_PATH)

    return model, scaler, le, df


# ============================================================
# SESSION STATE & AUTHENTICATION LOGIC
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "users_db" not in st.session_state:
    st.session_state.users_db = {
        "admin": {"password": "opticrop2026", "name": "Agronomy Admin"},
        "farmer": {"password": "crop123", "name": "Demo Farmer"}
    }
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def render_login_wall():
    st.title("🌾 OptiCrop")
    st.subheader("Human-Centered AI for Agricultural Decision Support & Crop Optimization")

    col_left, col_center, col_right = st.columns([1, 1.4, 1])
    with col_center:
        st.info(
            "👋 **Visiting Reviewer or Guest?** Jump straight into the interactive "
            "dashboard without typing credentials."
        )
        demo_label = "🚀 Instant 1-Click Demo Access (No Sign-In Required)"
        if st.button(demo_label, type="primary", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user_role = "Guest / Reviewer"
            st.session_state.username = "Guest Reviewer"
            st.rerun()

        st.divider()

        tab_login, tab_signup = st.tabs(["🔐 Registered Sign In", "📝 Create Account"])

        with tab_login:
            with st.form("login_form"):
                uname = st.text_input(
                    "Username",
                    value="admin",
                    help="Default demo account: admin / opticrop2026"
                )
                passwd = st.text_input("Password", type="password", value="opticrop2026")
                submit = st.form_submit_button("Sign In", use_container_width=True)
                if submit:
                    user_record = st.session_state.users_db.get(uname.strip())
                    if user_record and user_record["password"] == passwd:
                        st.session_state.authenticated = True
                        st.session_state.user_role = "Registered User"
                        st.session_state.username = user_record["name"]
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Try admin / opticrop2026 or 1-Click Demo Access.")

        with tab_signup:
            with st.form("signup_form"):
                new_name = st.text_input("Full Name", placeholder="e.g., Loukika Inkollu")
                new_user = st.text_input("Choose a Username")
                new_pass = st.text_input("Choose a Password", type="password")
                create_btn = st.form_submit_button("Register & Sign In", use_container_width=True)
                if create_btn:
                    if not new_user.strip() or not new_pass:
                        st.warning("Please enter both a username and password.")
                    elif new_user.strip() in st.session_state.users_db:
                        st.error("Username already exists. Please sign in.")
                    else:
                        display_name = new_name.strip() or new_user.strip()
                        st.session_state.users_db[new_user.strip()] = {
                            "password": new_pass,
                            "name": display_name
                        }
                        st.session_state.authenticated = True
                        st.session_state.user_role = "Registered User"
                        st.session_state.username = display_name
                        st.rerun()


# ============================================================
# MAIN DASHBOARD
# ============================================================
def render_dashboard():
    model, scaler, le, df = load_or_train_artifacts()
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    with st.sidebar:
        st.header("🌾 OptiCrop Advisor")
        st.caption(f"Signed in as: **{st.session_state.username}** ({st.session_state.user_role})")
        if st.button("🔓 Sign Out / Switch Account", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

        st.divider()
        nav = st.radio(
            "Navigation",
            [
                "🌱 Interactive Crop Advisor",
                "📊 Exploratory Data Analysis",
                "🏆 Model Benchmarks & Metrics",
                "📋 Dataset & Seasonal Explorer"
            ]
        )
        st.divider()
        st.markdown("**System Highlights**")
        st.markdown("- **Best Model:** Random Forest (99.55%)")
        st.markdown("- **Dataset:** 2,200 soil & climate samples")
        st.markdown("- **Crops Supported:** 22 varieties")
        st.markdown("- **Explainability:** Top-3 Confidence + Profile Match")

    # ------------------------------------------------------------
    # PAGE 1: INTERACTIVE CROP ADVISOR
    # ------------------------------------------------------------
    if nav == "🌱 Interactive Crop Advisor":
        st.title("🌱 Human-Centered Crop Recommendation & Decision Support")
        st.write(
            "Translate soil macronutrients and local weather conditions into explainable, "
            "confidence-calibrated crop recommendations."
        )

        st.subheader("⚡ Quick Regional Presets")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        if "preset" not in st.session_state:
            st.session_state.preset = {
                "N": 90, "P": 42, "K": 43, "temp": 21.0, "hum": 82.0, "ph": 6.5, "rain": 202.0
            }

        if p_col1.button("🌧️ Alluvial (Rice/Jute)", use_container_width=True):
            st.session_state.preset = {
                "N": 90, "P": 42, "K": 43, "temp": 21.0, "hum": 82.0, "ph": 6.5, "rain": 202.0
            }
        if p_col2.button("☀️ Black Soil (Cotton)", use_container_width=True):
            st.session_state.preset = {
                "N": 115, "P": 45, "K": 20, "temp": 24.5, "hum": 80.0, "ph": 6.8, "rain": 75.0
            }
        if p_col3.button("❄️ Dry Winter (Chickpea)", use_container_width=True):
            st.session_state.preset = {
                "N": 40, "P": 68, "K": 80, "temp": 18.5, "hum": 17.0, "ph": 7.3, "rain": 80.0
            }
        if p_col4.button("🌴 Tropical (Coffee/Fruit)", use_container_width=True):
            st.session_state.preset = {
                "N": 100, "P": 30, "K": 30, "temp": 25.5, "hum": 58.0, "ph": 6.8, "rain": 160.0
            }

        st.divider()
        col_inputs, col_results = st.columns([1.05, 1.25], gap="large")

        with col_inputs:
            st.subheader("🧪 Soil & Weather Parameters")
            c1, c2 = st.columns(2)
            with c1:
                n_val = st.slider("Nitrogen (N) [kg/ha]", 0, 140, int(st.session_state.preset["N"]))
                p_val = st.slider("Phosphorus (P) [kg/ha]", 5, 145, int(st.session_state.preset["P"]))
                k_val = st.slider("Potassium (K) [kg/ha]", 5, 205, int(st.session_state.preset["K"]))
                ph_val = st.slider("Soil pH Level", 3.5, 9.9, float(st.session_state.preset["ph"]), 0.1)
            with c2:
                temp_val = st.slider("Temperature (°C)", 8.0, 45.0, float(st.session_state.preset["temp"]), 0.5)
                hum_val = st.slider("Relative Humidity (%)", 14.0, 100.0, float(st.session_state.preset["hum"]), 0.5)
                rain_val = st.slider("Seasonal Rainfall (mm)", 20.0, 300.0, float(st.session_state.preset["rain"]), 1.0)

            save_btn = st.button("📌 Log Scenario to Session History", use_container_width=True)

        input_array = np.array([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]])
        input_scaled = scaler.transform(input_array)
        probs = model.predict_proba(input_scaled)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx] * 100

        best_crop = top3_crops[0]
        best_conf = top3_probs[0]
        best_season = SEASON_MAPPING.get(best_crop, "Perennial")

        if save_btn:
            st.session_state.prediction_history.append({
                "N": n_val, "P": p_val, "K": k_val,
                "Temp (°C)": temp_val, "Humidity (%)": hum_val,
                "pH": ph_val, "Rainfall (mm)": rain_val,
                "Recommended Crop": best_crop.title(),
                "Confidence (%)": f"{best_conf:.1f}%",
                "Season": best_season
            })
            st.toast(f"Saved {best_crop.title()} recommendation to session history!")

        with col_results:
            st.subheader("🎯 Explainable AI Recommendation")
            m1, m2, m3 = st.columns(3)
            m1.metric("Primary Crop", best_crop.upper())
            m2.metric("Model Confidence", f"{best_conf:.1f}%")
            m3.metric("Cropping Season", best_season.split()[0])

            st.markdown("#### Top-3 Candidate Crops (Probability Distribution)")
            for rank, (crop_name, conf) in enumerate(zip(top3_crops, top3_probs), start=1):
                season_tag = SEASON_MAPPING.get(crop_name, "Perennial")
                st.write(f"**#{rank} {crop_name.title()}** ({season_tag}) — `{conf:.1f}%`")
                st.progress(min(max(int(round(conf)), 1), 100))

            st.markdown(f"#### 🔍 Why **{best_crop.title()}**? (Your Input vs. Ideal Crop Profile)")
            crop_subset = df[df["label"] == best_crop][numeric_cols]
            ideal_means = crop_subset.mean()
            ideal_min = crop_subset.min()
            ideal_max = crop_subset.max()

            user_vals = [n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]
            comp_rows = []
            for feat, u_val in zip(numeric_cols, user_vals):
                in_range = ideal_min[feat] <= u_val <= ideal_max[feat]
                if in_range:
                    status = "✅ Optimal"
                elif u_val < ideal_min[feat]:
                    status = "⬇️ Below Typical"
                else:
                    status = "⬆️ Above Typical"
                comp_rows.append({
                    "Parameter": feat,
                    "Your Value": round(u_val, 2),
                    f"{best_crop.title()} Avg": round(ideal_means[feat], 2),
                    "Typical Range": f"{ideal_min[feat]:.1f} - {ideal_max[feat]:.1f}",
                    "Alignment": status
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        if st.session_state.prediction_history:
            st.divider()
            st.subheader("🗂️ Logged Session Scenarios")
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Session Recommendations (CSV)",
                data=csv_bytes,
                file_name="opticrop_recommendations.csv",
                mime="text/csv"
            )

    # ------------------------------------------------------------
    # PAGE 2: EXPLORATORY DATA ANALYSIS
    # ------------------------------------------------------------
    elif nav == "📊 Exploratory Data Analysis":
        st.title("📊 Exploratory Data Analysis (EDA) Gallery")
        st.write("Univariate, bivariate, and multivariate insights across 2,200 observations.")

        eda_tab1, eda_tab2, eda_tab3 = st.tabs([
            "📈 Univariate Distributions",
            "🔗 Bivariate Relationships",
            "🌐 Multivariate & PCA"
        ])

        with eda_tab1:
            c1, c2 = st.columns(2)
            p_hist = os.path.join(EDA_PLOTS_DIR, "univariate_histograms.png")
            p_box = os.path.join(EDA_PLOTS_DIR, "univariate_boxplots.png")
            p_dist = os.path.join(EDA_PLOTS_DIR, "univariate_crop_distribution.png")
            if os.path.exists(p_hist):
                c1.image(p_hist, caption="Feature Distributions with Mean & Median", use_container_width=True)
            if os.path.exists(p_box):
                c2.image(p_box, caption="Boxplots for IQR Outlier Detection", use_container_width=True)
            if os.path.exists(p_dist):
                st.image(p_dist, caption="Balanced Class Distribution (22 Crops)", use_container_width=True)

        with eda_tab2:
            p_scat = os.path.join(EDA_PLOTS_DIR, "bivariate_scatter_plots.png")
            p_crop_feat = os.path.join(EDA_PLOTS_DIR, "bivariate_crop_features.png")
            if os.path.exists(p_scat):
                st.image(p_scat, caption="Bivariate Feature Relationships", use_container_width=True)
            if os.path.exists(p_crop_feat):
                st.image(p_crop_feat, caption="Crop-Specific Nutrient Envelopes", use_container_width=True)

        with eda_tab3:
            c1, c2 = st.columns(2)
            p_heat = os.path.join(EDA_PLOTS_DIR, "multivariate_heatmap.png")
            p_pca = os.path.join(EDA_PLOTS_DIR, "multivariate_pca.png")
            p_imp = os.path.join(EDA_PLOTS_DIR, "multivariate_feature_importance.png")
            p_pair = os.path.join(EDA_PLOTS_DIR, "multivariate_pairplot.png")
            if os.path.exists(p_heat):
                c1.image(p_heat, caption="Feature Correlation Matrix", use_container_width=True)
            if os.path.exists(p_pca):
                c2.image(p_pca, caption="2D PCA Projection (74.4% Variance)", use_container_width=True)
            if os.path.exists(p_imp):
                st.image(p_imp, caption="Correlation with Target Crop Labels", use_container_width=True)
            if os.path.exists(p_pair):
                pair_label = "🔍 View Full High-Resolution Multivariate Pairplot"
                with st.expander(pair_label):
                    st.image(p_pair, caption="Pairplot of All 7 Agronomic Features", use_container_width=True)

    # ------------------------------------------------------------
    # PAGE 3: MODEL BENCHMARKS & METRICS
    # ------------------------------------------------------------
    elif nav == "🏆 Model Benchmarks & Metrics":
        st.title("🏆 Machine Learning Model Evaluation & Benchmarks")
        st.write("Comparative evaluation of 7 classifiers and K-Means clustering on 80/20 splits.")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Best Classifier", "Random Forest")
        m_col2.metric("Test Accuracy", "99.55%")
        m_col3.metric("Macro F1-Score", "1.00")
        m_col4.metric("Test Samples", "440 (20% Split)")

        c1, c2 = st.columns(2)
        p_comp = os.path.join(RESULTS_DIR, "model_comparison.png")
        p_fimp = os.path.join(RESULTS_DIR, "feature_importance.png")
        p_cm = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        p_elbow = os.path.join(RESULTS_DIR, "elbow_curve.png")

        if os.path.exists(p_comp):
            c1.image(p_comp, caption="Accuracy Comparison Across 7 Classifiers", use_container_width=True)
        if os.path.exists(p_fimp):
            c2.image(p_fimp, caption="Random Forest Gini Feature Importance", use_container_width=True)

        c3, c4 = st.columns([1.3, 1])
        if os.path.exists(p_cm):
            c3.image(p_cm, caption="Confusion Matrix on 440 Test Samples", use_container_width=True)
        if os.path.exists(p_elbow):
            c4.image(p_elbow, caption="K-Means Elbow Curve (k=5)", use_container_width=True)

        report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        if os.path.exists(report_path):
            rep_label = "📄 View Full Precision / Recall / F1 Classification Report"
            with st.expander(rep_label):
                with open(report_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="text")

    # ------------------------------------------------------------
    # PAGE 4: DATASET & SEASONAL EXPLORER
    # ------------------------------------------------------------
    elif nav == "📋 Dataset & Seasonal Explorer":
        st.title("📋 Agronomic Dataset & Seasonal Crop Explorer")
        col_f1, col_f2 = st.columns(2)
        all_seasons = sorted(df["Season"].unique())
        selected_seasons = col_f1.multiselect(
            "Filter by Cropping Season",
            options=all_seasons,
            default=all_seasons
        )
        available_crops = sorted(df[df["Season"].isin(selected_seasons)]["label"].unique())
        default_crops = available_crops[:6] if len(available_crops) > 6 else available_crops
        selected_crops = col_f2.multiselect(
            "Filter by Specific Crops",
            options=available_crops,
            default=default_crops
        )

        filtered_df = df[df["Season"].isin(selected_seasons)]
        if selected_crops:
            filtered_df = filtered_df[filtered_df["label"].isin(selected_crops)]

        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        st.subheader("📊 Mean Agronomic Profile by Selected Crop")
        if not filtered_df.empty:
            summary_df = (
                filtered_df.groupby(["label", "Season"])[numeric_cols]
                .mean()
                .round(2)
                .reset_index()
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ============================================================
# ENTRYPOINT ROUTER
# ============================================================
if not st.session_state.authenticated:
    render_login_wall()
else:
    render_dashboard()
