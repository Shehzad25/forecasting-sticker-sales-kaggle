# forecasting-sticker-sales-kaggle

Forecasting Sticker Sales
This repository contains the code for predicting future sticker sales based on historical data from the Kaggle competition "Forecasting Sticker Sales". The goal is to build a model that can accurately forecast the number of stickers sold, using features such as date, region, and category.

Project Structure
bash
Copy code
├── train.csv            # Training dataset
├── test.csv             # Test dataset
├── prediction.csv      # Final predictions
├── forecasting_sales.py # Main script for model training and predictions
Libraries and Dependencies
To run the code, the following libraries are required:

numpy
pandas
matplotlib
scikit-learn
xgboost
You can install the necessary dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Alternatively, install the required libraries manually:

bash
Copy code
pip install numpy pandas matplotlib scikit-learn==1.1.2 xgboost
Dataset
The project uses two datasets:

train.csv: The training dataset that contains historical data for sticker sales.
test.csv: The test dataset for which the model will predict the number of stickers sold.
Data Columns
id: Unique identifier for each record.
date: Date of the sale.
region: Region where the stickers were sold.
category: Category of the stickers.
num_sold: The number of stickers sold (target variable).
Code Overview
1. Data Loading and Exploration
The training and test datasets are loaded and basic exploration is performed, including displaying the first few rows, shape, data types, and summary statistics.

2. Handling Missing Values
Missing values in the num_sold column are filled with the mode (most frequent value) of the column.

3. Data Preprocessing
The id and date columns are dropped as they are not relevant for prediction.
Categorical variables (e.g., region, category) are encoded using LabelEncoder and OneHotEncoder.
4. Feature Scaling
The feature matrix is scaled using StandardScaler to standardize the features for improved model performance.

5. Model Training
The dataset is split into training and validation sets. The model used is XGBRegressor from the XGBoost library, which is trained on the training data.

6. Model Evaluation
The model's performance is evaluated using R-squared (r2_score) and cross-validation (cross_val_score).

7. Prediction
After training, the model predicts the number of stickers sold for the test dataset. The predictions are saved into a CSV file (prediction.csv), which includes the id and predicted num_sold.

Running the Code
To run the code and generate the predictions, simply execute the following:

bash
Copy code
python forecasting_sales.py
This will train the model, evaluate its performance, and generate the prediction file (prediction5.csv).

Notes
Ensure that the datasets (train.csv and test.csv) are present in the same directory as the script or adjust the file paths accordingly.
You may want to experiment with different machine learning models or hyperparameters to improve prediction accuracy.
Contact
For any questions, feel free to reach out via the https://www.kaggle.com/competitions/playground-series-s5e1.
