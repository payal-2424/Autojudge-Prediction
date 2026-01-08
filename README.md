## AutoJudge – Programming Problem Difficulty Prediction System

AutoJudge is a machine learning–based system that predicts the difficulty level of programming problems using only textual information.
It provides both:

Difficulty class → Easy / Medium / Hard

Numerical difficulty score → continuous value

The system includes a Streamlit web interface for interactive usage.

 # Features:

1)  Text-based difficulty prediction (no code required)
2)  Difficulty classification (Easy / Medium / Hard)
3)  Difficulty regression score
4)  Machine learning using Extra Trees & Random Forest
5)  Simple and intuitive Streamlit Web UI
6) Modular and deployable architecture

# Machine Learning Approach
1. Feature Engineering

Combined textual fields:

Problem title

Problem description

Input description

Output description

Sample input/output (if available)

TF-IDF vectorization for text

Handcrafted features:

Character count

Word count

Digit count

Math symbol frequency

Keyword-based indicators (graph, dp, tree, greedy, etc.)

2. Models Used
Task	Model
Classification	Random Forest Classifier
Regression	Extra Trees Regressor

Both models are trained using scikit-learn pipelines.

# Model Performance
🔹 Classification

Accuracy: 53%

🔹 Regression

RMSE: 0.922

MAE: 0.757

These results demonstrate effective learning using only textual problem descriptions.

# Web Interface

Built using Streamlit

Accepts:

Problem title

Problem description

Input format

Output format

Displays:

Predicted difficulty class

Predicted numerical score

# Project Structure
AutoJudge/
│
├── app.py                     # Streamlit web app
├── training_notebook.ipynb    # Model training & evaluation
├── requirements.txt
├── README.md
│
└── models/
    ├── autojudge_classifier.pkl
    └── autojudge_regressor.pkl

 Model Files 

⚠️ The trained .pkl model files are large in size and exceed GitHub upload limits.

 Model Download Link-  
 autojudge_regressor- https://drive.google.com/file/d/1ujg0OeA28QuTh5mz8Xxa2wLaX003gfFF/view?usp=sharing
 autojudge_classifier- https://drive.google.com/file/d/1VGEwBeaKgoAf3X1uXr9TACSRv_ONUx-p/view?usp=sharing


After downloading:

Extract the files

Place them in the same directory as app.py

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit App
streamlit run app.py


🛠️ Technologies Used

Python 3

scikit-learn

pandas, numpy

Streamlit

Pickle / Joblib


Notes:

The regression model is trained on original difficulty scores (no normalization at inference time)

Model files are loaded dynamically during runtime

Designed for academic submission and deployment

# Conclusion

AutoJudge demonstrates that programming problem difficulty can be effectively predicted using only textual information, achieving reliable classification and regression performance through machine learning and feature engineering.
