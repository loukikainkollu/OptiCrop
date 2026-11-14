# 🌾 OptiCrop: Human-Centered AI Crop Advisor

**OptiCrop** is an interactive agricultural production optimization and decision-support application built with Python, Scikit-Learn, and Streamlit. It translates soil macronutrients (`N`, `P`, `K`, `pH`) and regional weather conditions (`temperature`, `humidity`, `rainfall`) into confidence-calibrated crop recommendations with human-centered explainability.

## ✨ Key Features
- **Authentication & Instant Reviewer Demo Mode:** Includes a full login/registration wall alongside a **1-Click Instant Demo Access** button so reviewers can test the live system immediately without sign-in friction.
- **Explainable Crop Recommendation Engine:** Powered by a **Random Forest Classifier (99.55% test accuracy)** trained on 2,200 agronomic samples across 22 crops. Outputs Top-3 probability bars and a side-by-side comparison of user inputs against the recommended crop's ideal historical profile.
- **Interactive Regional Presets & Session Logging:** Test alluvial, semi-arid, winter, and perennial profiles with a single click and export logged scenarios to CSV.
- **Complete EDA & Model Benchmark Suite:** Browse univariate, bivariate, PCA, K-Means clustering, and confusion matrix visualizations directly inside the app.

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
