# Data Science Project: Bank Fraud Detection

This data science project focuses on building models to detect bank fraud using a synthetic dataset from a financial payment system. The goal is to develop accurate classifiers that can effectively distinguish fraudulent transactions from non-fraudulent ones.

## Dataset

The dataset used in this project is sourced from a financial payment system and contains information about various features related to transactions. The dataset includes the following columns:

- label: The target variable indicating whether a transaction is fraudulent (1) or non-fraudulent (0).
- msisdn: Mobile Subscriber ISDN Number.
- aon: Age on network.
- daily_decr30: Average daily amount spent in the last 30 days.
- daily_decr90: Average daily amount spent in the last 90 days.
- rental30: Average rent amount in the last 30 days.
- rental90: Average rent amount in the last 90 days.
- last_rech_date_ma: Number of days since the last main account recharge.
- last_rech_date_da: Number of days since the last data account recharge.
- last_rech_amt_ma: Amount of the last main account recharge.
- cnt_ma_rech30: Number of times main account got recharged in the last 30 days.
- fr_ma_rech30: Frequency of main account recharged in the last 30 days.
- sumamnt_ma_rech30: Total amount of main account recharged in the last 30 days.
- medianamnt_ma_rech30: Median amount of main account recharged in the last 30 days.
- medianmarechprebal30: Median main account recharge prebalance in the last 30 days.
- cnt_ma_rech90: Number of times main account got recharged in the last 90 days.
- fr_ma_rech90: Frequency of main account recharged in the last 90 days.
- sumamnt_ma_rech90: Total amount of main account recharged in the last 90 days.
- medianamnt_ma_rech90: Median amount of main account recharged in the last 90 days.
- medianmarechprebal90: Median main account recharge prebalance in the last 90 days.
- cnt_da_rech30: Number of times data account got recharged in the last 30 days.
- fr_da_rech30: Frequency of data account recharged in the last 30 days.
- cnt_da_rech90: Number of times data account got recharged in the last 90 days.
- fr_da_rech90: Frequency of data account recharged in the last 90 days.
- cnt_loans30: Number of loans taken by the user in the last 30 days.
- amnt_loans30: Total amount of loans taken by the user in the last 30 days.
- maxamnt_loans30: Maximum amount of loans taken by the user in the last 30 days.
- medianamnt_loans30: Median of amounts of loans taken by the user in the last 30 days.
- cnt_loans90: Number of loans taken by the user in the last 90 days.
- amnt_loans90: Total amount of loans taken by the user in the last 90 days.
- maxamnt_loans90: Maximum amount of loans taken by the user in the last 90 days.
- medianamnt_loans90: Median of amounts of loans taken by the user in the last 90 days.
- payback30: Average payback time in days over the last 30 days.
- payback90: Average payback time in days over the last 90 days.
- pcircle: Telecom circle.
- pdate: Date.

## Project Structure

The project consists of the following files:

- `data.csv`: The dataset used for training and evaluation.
- `bank_fraud_detection.ipynb`: Jupyter Notebook containing the data preprocessing, model training, and evaluation code.
- `README.md`: This file, providing an overview of the project.

## Project Workflow

1. Data Exploration and Visualization: Perform exploratory data analysis to gain insights into the dataset. Visualize the distribution of features and the class imbalance.
2. Data Preprocessing: Preprocess the data by handling missing values, converting categorical features into numerical representations, and performing feature scaling if necessary.
3. Model Training: Train different classifiers such as K-Nearest Neighbors (KNN), Random Forest, and XGBoost on the preprocessed data.
4. Model Evaluation: Evaluate the trained models using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score. Also, plot the confusion matrix and ROC curves to assess the performance of the models.
5. Ensemble Model: Create an ensemble model using Voting Classifier to combine the predictions of multiple classifiers and improve overall performance.
6. Conclusion: Summarize the results, discuss the performance of the models, and provide insights for future improvements.

## Getting Started

To run the code in this project, follow these steps:

1. Install the required libraries and dependencies mentioned in the `bank_fraud_detection.ipynb` notebook.
2. Download the `data.csv` file and place it in the same directory as the notebook.
3. Open the `bank_fraud_detection.ipynb` notebook using Jupyter Notebook or any compatible environment.
4. Execute the cells in the notebook sequentially to perform data preprocessing, model training, and evaluation.

## Results

The project utilizes K-Nearest Neighbors, Random Forest, XGBoost, and an ensemble model for bank fraud detection. Here are the key results:

- K-Nearest Neighbors achieved an accuracy of 89% with precision, recall, and F1-score of 0.91, 0.97, and 0.94, respectively.
- Random Forest Classifier achieved an accuracy of 78% with precision, recall, and F1-score of 0.96, 0.78, and 0.86, respectively.
- XGBoost Classifier encountered an error during training, so the results are not available.
- The ensemble model combining KNN, Random Forest, and XGBoost achieved an accuracy of 89% with precision, recall, and F1-score of 0.91, 0.97, and 0.94, respectively.

## Conclusion

The project demonstrates the application of machine learning techniques to detect bank fraud using a synthetic dataset. The ensemble model, combining multiple classifiers, shows promising results in accurately identifying fraudulent transactions. However, further optimization and fine-tuning of the models can be explored to improve performance.

For more details, refer to the `main.ipynb` notebook.

