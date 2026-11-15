import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_prepare_data, get_full_dataset
from src.generative_classifier import GaussianGenerativeClassifier
from src.evaluation import evaluate_model
from src.visualizations import plot_confusion_matrix, plot_posterior_probabilities
from src.utils import explain_prediction

st.set_page_config(
    page_title="Iris Probabilistic Generative Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED PREMIUM CSS (BRANDKIT TASTE) ---
st.markdown("""
<style>
    /* Dark Premium Canvas */
    .stApp {
        background-color: #0d0f12;
        color: #e5e7eb;
        font-family: 'Inter', 'SF Pro Display', sans-serif;
    }
    
    /* Sparse, Cinematic Headers */
    h1, h2, h3 {
        font-weight: 300 !important;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    h1 {
        font-size: 2.5rem !important;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 0.5rem;
        margin-bottom: 2rem;
    }
    
    /* Clean grid / panel look */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 300;
        color: #10b981;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem;
    }
    
    /* Premium card containers */
    .stDataFrame {
        border-radius: 4px;
        border: 1px solid #1f2937;
        overflow: hidden;
    }
    
    /* Sidebar restyling */
    [data-testid="stSidebar"] {
        background-color: #080a0c;
        border-right: 1px solid #1f2937;
    }
    
    /* Button styling (accent color) */
    .stButton>button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Elegant Dividers */
    hr {
        border-color: #1f2937;
    }
    
    /* Text formatting */
    p, li {
        color: #d1d5db;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED DATA & MODEL ---
@st.cache_data
def get_data():
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()
    return X_train, X_test, y_train, y_test, feature_names, target_names

@st.cache_resource
def get_model(X_train, y_train):
    model = GaussianGenerativeClassifier()
    model.fit(X_train, y_train)
    return model

X_train, X_test, y_train, y_test, feature_names, target_names = get_data()
model = get_model(X_train, y_train)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Class Statistics", "Model Mathematics", "Prediction", "Evaluation", "About"])

# --- HOME ---
if page == "Home":
    st.title("Iris Probabilistic Generative Classifier")
    st.markdown("""
    <div style="margin-bottom: 3rem;">
        <h3 style="color: #9ca3af; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: -1rem;">Gaussian Class-Conditional Density + Bayes Posterior</h3>
    </div>
    
    **Classify Iris flowers by modeling each class as a multivariate Gaussian distribution and applying Bayes' theorem.**
    
    <div style="background-color: #111827; padding: 2rem; border-radius: 4px; border: 1px solid #1f2937; margin-top: 2rem;">
        <h4 style="margin-top: 0; color: #60a5fa; font-weight: 400;">PIPELINE ARCHITECTURE</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">01</span><br/>Dataset Features</div>
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">02</span><br/>Class-wise Mean <span style="color:#60a5fa">μ</span></div>
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">03</span><br/>Covariance Matrix <span style="color:#60a5fa">Σ</span></div>
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">04</span><br/>Gaussian Likelihood</div>
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">05</span><br/>Class Prior <span style="color:#60a5fa">P(C)</span></div>
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">06</span><br/>Bayes Theorem</div>
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">07</span><br/>Posterior Probability</div>
            <div style="border-left: 2px solid #374151; padding-left: 1rem;"><span style="color: #9ca3af; font-size: 0.8rem;">08</span><br/><span style="color: #10b981;">Predicted Class</span></div>
        </div>
    </div>
    
    <div style="margin-top: 3rem;">
        <h4 style="color: #9ca3af; font-weight: 400;">SYSTEM SPECIFICATIONS</h4>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin-bottom: 0.5rem;"><span style="color: #6b7280; width: 150px; display: inline-block;">Features:</span> Sepal Length, Sepal Width, Petal Length, Petal Width</li>
            <li style="margin-bottom: 0.5rem;"><span style="color: #6b7280; width: 150px; display: inline-block;">Classes:</span> Setosa, Versicolor, Virginica</li>
            <li style="margin-bottom: 0.5rem;"><span style="color: #6b7280; width: 150px; display: inline-block;">Architecture:</span> Probabilistic Generative Classifier</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- DATASET ---
elif page == "Dataset":
    st.title("Dataset Overview")
    
    X_full, y_full, f_names, t_names = get_full_dataset()
    df = pd.DataFrame(X_full, columns=f_names)
    df['Target Class'] = [t_names[y] for y in y_full]
    
    st.write(f"**Total Samples:** {len(X_full)}")
    st.write(f"**Number of Features:** {len(f_names)}")
    st.write("**Classes:**", ", ".join(t_names))
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    st.subheader("Sample Data (First 10 Rows)")
    st.dataframe(df.head(10))
    
    st.subheader("Training/Testing Split")
    st.write(f"- Training set: {X_train.shape[0]} samples")
    st.write(f"- Testing set: {X_test.shape[0]} samples")

# --- CLASS STATISTICS ---
elif page == "Class Statistics":
    st.title("Class Statistics")
    
    for c in model.classes_:
        class_name = target_names[c].upper()
        st.subheader(f"Class: {class_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prior Probability P(C)", f"{model.priors_[c]:.4f}")
            st.markdown("**Mean Vector $\mu$:**")
            mean_df = pd.DataFrame([model.means_[c]], columns=feature_names)
            st.dataframe(mean_df.style.format("{:.4f}"))
            
        with col2:
            st.markdown("**Covariance Matrix $\Sigma$:**")
            cov_df = pd.DataFrame(model.covariances_[c], columns=feature_names, index=feature_names)
            st.dataframe(cov_df.style.format("{:.4f}"))
        
        st.divider()

# --- MODEL MATHEMATICS ---
elif page == "Model Mathematics":
    st.title("Model Mathematics")
    
    st.markdown("""
    ### 1. Prior Probability
    The prior probability of a class $C_k$ is the fraction of training samples that belong to it:
    $$P(C_k) = \\frac{N_k}{N}$$
    
    ### 2. Mean Vector
    The mean vector $\mu_k$ for class $C_k$ is the average of the features of its samples:
    $$\mu_k = \\frac{1}{N_k} \sum_{i=1}^{N_k} x_i$$
    
    ### 3. Covariance Matrix
    Using Maximum Likelihood Estimation, the covariance matrix $\Sigma_k$ is:
    $$\Sigma_k = \\frac{1}{N_k} \sum_{i=1}^{N_k} (x_i - \mu_k)(x_i - \mu_k)^T$$
    
    ### 4. Gaussian PDF (Likelihood)
    The class-conditional density is modeled as a multivariate Gaussian:
    $$P(x | C_k) = \\frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \\right)$$
    
    ### 5. Bayes Theorem
    $$P(C_k | x) = \\frac{P(x | C_k) P(C_k)}{\sum_{j} P(x | C_j) P(C_j)}$$
    Where $P(C_k | x)$ means: "Probability that the flower belongs to class C after observing its measurements."
    
    ### 6. Discriminant Function
    A log discriminant score used for numerical stability:
    $$g_k(x) = -\\frac{1}{2} \ln |\Sigma_k| - \\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \ln P(C_k)$$
    
    ### 7. Final Decision Rule
    Predict the class with the maximum posterior probability (or discriminant score):
    $$\hat{C} = \mathrm{argmax}_k P(C_k | x) = \mathrm{argmax}_k g_k(x)$$
    """)

# --- PREDICTION ---
elif page == "Prediction":
    st.title("Live Prediction")
    
    st.sidebar.subheader("Input Measurements")
    sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 6.0, 0.1)
    sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 2.9, 0.1)
    pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.5, 0.1)
    pw = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.5, 0.1)
    
    x_unknown = np.array([sl, sw, pl, pw])
    
    st.write(f"**Input Vector $x$:** `[{sl}, {sw}, {pl}, {pw}]`")
    
    if st.button("Classify Flower", type="primary"):
        # Calculate
        log_likelihoods = [model._log_gaussian_density(x_unknown.reshape(1, -1), model.means_[c], model.covariances_[c])[0] for c in model.classes_]
        likelihoods = np.exp(log_likelihoods)
        priors = [model.priors_[c] for c in model.classes_]
        posteriors = model.predict_proba(x_unknown.reshape(1, -1))[0]
        scores = model.decision_function(x_unknown.reshape(1, -1))[0]
        
        pred_idx = np.argmax(posteriors)
        pred_class = target_names[pred_idx]
        
        st.markdown(f"""
        <div style="background-color: #111827; border-left: 4px solid #10b981; padding: 1.5rem; margin-top: 2rem; margin-bottom: 2rem;">
            <p style="margin: 0; color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em;">Predicted Class</p>
            <h2 style="margin: 0; color: #10b981; font-size: 2.5rem; border: none; padding: 0;">{pred_class.upper()}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Results Table
        results_data = {
            "Class": [target_names[c].capitalize() for c in model.classes_],
            "Prior": [f"{p:.4f}" for p in priors],
            "Likelihood": [f"{l:.4e}" for l in likelihoods],
            "Posterior": [f"{p:.4f}" for p in posteriors],
            "Discriminant Score": [f"{s:.4f}" for s in scores]
        }
        st.table(pd.DataFrame(results_data))
        
        # Posterior Chart
        st.subheader("Posterior Probabilities")
        fig = plot_posterior_probabilities(posteriors, target_names)
        st.pyplot(fig)
        
        with st.expander("Show Step-by-Step Explanation"):
            st.text(explain_prediction(x_unknown, model, target_names))

# --- EVALUATION ---
elif page == "Evaluation":
    st.title("Model Evaluation")
    
    eval_results = evaluate_model(model, X_test, y_test, target_names)
    
    st.metric("Test Set Accuracy", f"{eval_results['accuracy'] * 100:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(eval_results['confusion_matrix'], target_names)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Classification Report")
        st.text(eval_results['classification_report'])

# --- ABOUT ---
elif page == "About":
    st.title("About")
    st.write("""
    This project was built for an academic Machine Learning course.
    It demonstrates a manual implementation of a Probabilistic Generative Classifier using Multivariate Gaussian Distributions and Bayes Theorem.
    
    No black-box scikit-learn models (like `GaussianNB`) were used for the core classification logic. The mean vectors, covariance matrices, likelihoods, priors, and posteriors are manually computed using NumPy.
    """)
