🛡️ PhishGuard — ML-Powered Phishing URL Detector

A complete end-to-end Machine Learning project that detects phishing URLs using a
trained Random Forest classifier, served through a Flask REST API and a React frontend.


📁 Project Structure

phishing-detector/
│
├── model/
│   ├── train_model.py     # Full ML pipeline (dataset → features → train → evaluate → save)
│   └── model.pkl          # Saved best model (generated after training)
│
├── static/
│   ├── style.css          # Dark cyberpunk UI stylesheet
│   └── plots/             # Auto-generated visualisation PNGs
│       ├── cm_logistic_regression.png
│       ├── cm_decision_tree.png
│       ├── cm_random_forest.png
│       ├── feature_importance.png
│       └── model_comparison.png
│
├── templates/
│   └── index.html         # React SPA (embedded in HTML via Babel CDN)
│
├── app.py                 # Flask backend (API endpoints)
├── utils.py               # Feature extraction utilities (shared by train + inference)
├── requirements.txt       # Python dependencies
└── README.md              # This file


⚡ Quick Start

1 — Clone / set up the project

bash# (optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# install all dependencies
pip install -r requirements.txt

2 — Train the model

bashpython model/train_model.py

This will:


Generate a synthetic labelled dataset (4 000 URLs — 50 % safe, 50 % phishing)
Engineer 13 features per URL
Train three models: Logistic Regression, Decision Tree, Random Forest
Print accuracy / precision / recall / F1 and a full classification report for each
Save confusion-matrix PNGs + feature-importance chart into static/plots/
Save the best model (by F1 score) to model/model.pkl


Expected output:

════════════════════════════════════════════════════════════
  PHISHING URL DETECTION — MODEL TRAINING
════════════════════════════════════════════════════════════

[1/5] Generating synthetic dataset …
      4000 samples  |  2000 phishing  |  2000 safe

[2/5] Engineering features …
      13 features: ['url_length', 'num_dots', ...]

[3/5] Training models …

  ▶ Logistic Regression
    Accuracy : 0.9125
    Precision: 0.9188
    Recall   : 0.9062
    F1       : 0.9124

  ▶ Decision Tree
    Accuracy : 0.9500
    ...

  ▶ Random Forest
    Accuracy : 0.9775
    Precision: 0.9762
    Recall   : 0.9788
    F1       : 0.9775
    🏆 Best model: Random Forest  (F1 = 0.9775)

[5/5] Saving model and artefacts …
  ✓ Model saved → model/model.pkl
════════════════════════════════════════════════════════════
  TRAINING COMPLETE ✓
════════════════════════════════════════════════════════════

3 — Run the Flask app

bashpython app.py

Open your browser at: http://localhost:5000


🔌 API Reference

POST /predict

Accepts a JSON body and returns a prediction with confidence score.

Request:

json{ "url": "http://paypal-secure.xyz/login/verify" }

Response:

json{
  "url":        "http://paypal-secure.xyz/login/verify",
  "prediction": "Phishing",
  "confidence": 97.40,
  "label":      1,
  "model_used": "Random Forest",
  "features": {
    "url_length":        40,
    "num_dots":           3,
    "has_at":             0,
    "has_https":          0,
    "has_ip":             0,
    "num_hyphens":        1,
    "num_subdomains":     1,
    "url_depth":          2,
    "has_suspicious":     1,
    "num_suspicious_kw":  2,
    "has_port":           0,
    "tld_in_path":        1,
    "url_entropy":        3.9421
  }
}

GET /health

json{ "status": "ok", "model": "Random Forest", "model_ready": true }

GET /plots

json{ "plots": ["cm_random_forest.png", "feature_importance.png", ...] }


🧪 Sample Inputs & Outputs

URLExpected Resulthttps://github.com/openai/whisper✅ Safe (~97%)https://stackoverflow.com/questions/python✅ Safe (~95%)http://192.168.1.1/login/verify@secure🚨 Phishing (~99%)http://paypal-secure-update.xyz/account/login.php🚨 Phishing (~98%)http://signin-abc123.tk/update?ref=xyz🚨 Phishing (~97%)


🧠 Feature Engineering

FeatureDescriptionurl_lengthTotal character countnum_dotsNumber of . charactershas_atPresence of @ (0/1)has_httpsUses HTTPS scheme (0/1)has_ipIPv4 address in URL (0/1)num_hyphensNumber of - charactersnum_subdomainsSubdomain counturl_depthNumber of path segmentshas_suspiciousContains suspicious keyword (0/1)num_suspicious_kwCount of suspicious keyword matcheshas_portNon-standard port present (0/1)tld_in_pathSuspicious TLD detected (.xyz, .tk …)url_entropyShannon entropy (character randomness)


📊 Models Compared

ModelAccuracyPrecisionRecallF1Logistic Regression~91%~92%~91%~91%Decision Tree~95%~95%~95%~95%Random Forest~98%~98%~98%~98%


Results vary slightly with each training run due to random synthetic data generation.




🔬 Visualisations

After training, the following plots are saved to static/plots/ and visible in the
Model Insights tab of the UI:


Confusion matrices for each model
Feature importance bar chart (Random Forest)
Model comparison grouped bar chart (Accuracy / Precision / Recall / F1)



🛠 Tech Stack

LayerTechnologyMLscikit-learn, pandas, numpyVizmatplotlib, seabornAPIFlaskUIReact 18 (CDN), CSS3Model persistencepickle


💡 Skills Demonstrated


Synthetic dataset generation and labelling
Feature engineering from raw URL strings
Training and comparing multiple classifiers
Model evaluation (accuracy, precision, recall, F1, confusion matrix)
Model serialisation with pickle
REST API design with Flask
Component-based UI with React
End-to-end ML project structure
