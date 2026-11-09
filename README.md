🏏 IPLytics — IPL Match Winner Predictor



IPLytics is a machine learning-based IPL match winner prediction project built using Python and Streamlit. It analyzes historical IPL match data and estimates the probability of either team winning an upcoming match.



🚀 Project Overview



IPLytics uses historical match statistics and machine learning algorithms to generate match predictions through an interactive dashboard.



The project compares multiple machine learning models and selects a final model based on validation performance.



✨ Features



Predict the winner of an IPL match.



Compare winning probabilities for both teams.



Select teams and match venue through an interactive dashboard.



Use historical team performance and recent form.



Include head-to-head, venue and Elo-based features.



Incorporate estimated squad-related features.



Compare Logistic Regression, Random Forest and XGBoost.



Display predictions through a Streamlit dashboard.



🤖 Machine Learning Models



The project evaluates three algorithms:



Logistic Regression



Random Forest



XGBoost



Selected model: Logistic Regression, based on the recent-season validation comparison.



The model selection process is designed to compare performance rather than assume that a more complex algorithm will always perform better.



🛠️ Technology Stack



Technology



Purpose



Python



Programming language



Pandas



Data processing



NumPy



Numerical operations



Scikit-learn



Machine learning



XGBoost



Model comparison



Matplotlib



Data visualization



Joblib



Model serialization



Streamlit



Interactive dashboard



📁 Project Structure



IPLytics/

│

├── app/

│   └── app.py

│

├── data/

│   ├── processed/

│   │   ├── matches.csv

│   │   ├── features.csv

│   │   └── player\_features.csv

│   │

│   └── raw/

│

├── models/

│   ├── logistic\_regression\_final.joblib

│   ├── random\_forest\_final.joblib

│   ├── xgboost\_final.joblib

│   ├── model\_selection.json

│   └── algorithm\_comparison\_results.csv

│

├── reports/

│   └── Model evaluation and analysis reports

│

├── src/

│   ├── data\_preprocessing.py

│   ├── feature\_engineering.py

│   ├── player\_features.py

│   ├── train.py

│   ├── train\_final.py

│   └── predict.py

│

├── tests/

│

├── .streamlit/

│   └── config.toml

│

├── .gitignore

├── requirements.txt

└── README.md



⚙️ Installation and Setup



1\. Clone the repository



git clone https://github.com/Ayush-1242/IPLytics.git



2\. Navigate to the project directory



cd IPLytics



3\. Create a virtual environment



python -m venv .venv



4\. Activate the virtual environment



Windows CMD:



.venv\\Scripts\\activate



5\. Install dependencies



pip install -r requirements.txt



6\. Run the application



streamlit run app/app.py



The application will open in your browser.



📊 Model Evaluation



IPLytics compares multiple machine learning algorithms using historical match data and recent-season validation.



Model performance is evaluated using classification metrics and additional validation experiments.



The selected model is Logistic Regression.



Important: Predictions are experimental estimates. Validation performance is limited, so the displayed probabilities should not be interpreted as guaranteed outcomes or used as betting advice.



🎯 Project Objective



The objective of IPLytics is to explore how historical cricket data, feature engineering and machine learning can be combined to estimate IPL match outcomes through an interactive application.

