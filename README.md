# **Email Campaign Optimization: Maximizing Click-Through Rate (CTR) using XGBoost and SHAP**
This project demonstrates a full pipeline to maximize email link click-through rates (CTR) through machine learning, using user and email metadata. It applies binary classification techniques to predict email opens (EO) and link clicks (LO), and strategically combines their probabilities to optimize campaign outcomes.

# **Objectives**
Predict whether a user will open an email (EO).

Predict whether a user will click on a link inside the email (LO), conditional on EO.

Combine EO and LO probabilities to derive a final conversion score.

Rank users to target the most promising ones.

Compare Random vs Optimized strategies using A/B testing.

# **Data Sources**
Three CSV datasets:

email_opened_table.csv: Contains email_id values where the email was opened.

email_table.csv: Main dataset with email metadata.

link_clicked_table.csv: Contains email_id values where links inside the email were clicked.

  # **Data Preprocessing**
Feature Engineering:

Created two new binary columns:

EO: 1 if email was opened, 0 otherwise.

LO: 1 if link was clicked, 0 otherwise.

  # **Label Encoding**

Applied to categorical features: email_text, email_version, user_country, weekday.

  # **Correlation Heatmap**

Visual analysis to explore feature relationships.

**Stage 1: Email Open Prediction (EO)**
Feature-Target Split:

Features: all except email_id, EO, LO, and email_version.

Target: EO.

Train-Test Split (80/20)

Class Imbalance Handling:

Used SMOTETomek to balance classes in the training set.

Model Training with XGBoost (GPU Accelerated):

Performed RandomizedSearchCV for hyperparameter tuning.

Trained using logloss and evaluated with accuracy.

Explainability:

Used SHAP (SHapley Additive exPlanations) to interpret feature importance.

**Stage 2: Link Click Prediction (LO | EO=1)**
Filtered Subset: Only used rows where EO == 1.

Train-Test Split & Model Training:

Trained a separate XGBoost classifier to predict LO.

Evaluation:

Classification report, ROC-AUC score, confusion matrix.

Explainability:

Applied SHAP again for insight into the most influential features.

Combined Conversion Probability
Final Probability = EO_Prob * LO_Prob

Used this combined score to rank users by likelihood of clicking.

# **Performance Evaluation**
Top-K Strategy
Selected top 20% of users based on final conversion score.

Compared Click-Through Rates (CTR):

Baseline CTR (all users)

**Top-K CTR (predicted high-probability users)**

**Lift Curve**
Evaluated model lift over random chance by plotting cumulative clicks vs proportion of users contacted.

**A/B Test Simulation**
Simulated a campaign:

Group A: Randomly selected users.

Group B: Top K users from model predictions.

Compared CTR between groups via bar plot.

# **Key Libraries Used**
pandas, numpy for data manipulation

matplotlib, seaborn for visualization

xgboost for model building

sklearn for preprocessing & evaluation

imblearn for SMOTETomek resampling

shap for model explainability

# **Results Summary**
Model-based ranking significantly improved CTR in Group B compared to random selection.

SHAP plots provided transparency into what drives opens and clicks.

The final conversion probability offers a robust way to prioritize user outreach for email campaigns.
