# Credit Card Fraud Detection Using Machine Learning

## Overview
This project aims to develop a machine learning model to accurately detect fraudulent credit card transactions. By leveraging various algorithms and techniques, the model helps in identifying potentially fraudulent activities, providing valuable insights for fraud prevention.

## Dataset
The dataset used for this project is sourced from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This dataset contains transactions made by European cardholders in September 2013. It includes 284,807 transactions, out of which 492 are fraudulent. The dataset features numerical input variables that are the result of a PCA transformation and contains only the amount and time features of the original dataset.

## Objectives
- **Detect fraudulent transactions:** Develop a model that accurately identifies fraudulent credit card transactions.
- **Optimize model performance:** Experiment with various machine learning algorithms and techniques to maximize detection accuracy and minimize false positives.
- **Provide insights:** Analyze the data to uncover patterns and insights that can help in understanding and preventing fraud.

## Methodology
1. **Data Collection and Preprocessing:**
   - Imported the dataset from Kaggle.
   - Handled missing values and performed data normalization.
   - Applied PCA transformation to reduce dimensionality and retain important features.

2. **Exploratory Data Analysis (EDA):**
   - Analyzed the distribution of classes (fraudulent vs. non-fraudulent) to understand the class imbalance.
   - Visualized the data using histograms, box plots, and scatter plots to identify patterns and anomalies.

3. **Model Development:**
   - Split the data into training and testing sets.
   - Trained various models, including Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting.
   - Used cross-validation to tune hyperparameters and avoid overfitting.

4. **Model Evaluation:**
   - Evaluated model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
   - Plotted confusion matrices and ROC curves to visualize model performance.
   - Selected the best-performing model based on evaluation metrics.

## Results
- **Model Performance:** Achieved high accuracy and recall in detecting fraudulent transactions with minimal false positives.
- **Insights:** Identified key features contributing to fraud detection, providing valuable insights for fraud prevention strategies.

### Confusion Matrix
The confusion matrix below illustrates the performance of our model in terms of true positives, true negatives, false positives, and false negatives.

![image](https://github.com/user-attachments/assets/88680f0e-80c4-4cb1-b123-ac54bd34a6e5)



### AUROC Curve
The Area Under the Receiver Operating Characteristic (AUROC) curve below shows the model's ability to distinguish between fraudulent and non-fraudulent transactions.

![image](https://github.com/user-attachments/assets/71361889-0acb-4798-9f89-3c01f8368e71)


## Usage
To use the model, follow these steps:
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the notebook `Credit_Card_Fraud_Detection.ipynb` to see the full analysis and model training process.
4. Use the trained model to predict fraudulent transactions on new data.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Kaggle for providing the dataset.
- All contributors and the open-source community.
