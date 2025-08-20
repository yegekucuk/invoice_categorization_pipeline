**Invoice Categorization Pipeline**
===================================

Categorizing invoices using machine learning with scikit-learn pipelines.

### Description

`Invoice Categorization Pipeline` builds a machine learning model to classify invoices based on numeric, categorical, and text features.  
It preprocesses data, trains a `LogisticRegression` model, evaluates its performance, and saves the trained model.

### Features

* Load and preprocess invoice data from CSV
* Encode categorical and text data (`OneHotEncoder`, `CountVectorizer`)
* Apply scaling with `StandartScaler`
* Train a Logistic Regression classifier
* Evaluate model performance (accuracy, classification report, confusion matrix)
* Save the trained model as a `.pkl` file
