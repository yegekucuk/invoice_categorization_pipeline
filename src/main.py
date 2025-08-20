import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from functions import *

# Paths
DATA_PATH, RESULTS_PATH= "data", "results"

# Import Pandas DataFrame
df = pd.read_csv(f"{DATA_PATH}/data.csv")

# Create Y output
le = LabelEncoder()
y = le.fit_transform(df["category"])

# Create X input
X = df.drop(columns=["category", "invoice_id", "date", "vat_percent", "currency", "paid"])

# Classifying the columns for Sklearn Pipeline
numeric_features = ["net_amount"]
categorical_features = ["vendor", "payment_method"]
text_features = "description"

# Transformers
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
text_transformer = CountVectorizer()

# Creating Feature Engineering pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_features)
    ]
)

# Creating model pipeline
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=10000, random_state=42))
    ]
)

# Train-test split, DONT maintain the classes percentage distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None, test_size=0.3, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make a predict and evaluate
y_pred = model.predict(X_test)
accuracy, report, cm = evaluate_model(y_test, y_pred)

# Print classification report
print(f"Classification Report\n{report}")

# Draw confusion matrix
draw_confusion_matrix(cm, save_plot=True)

# Save model
save_model(model, path=f"{RESULTS_PATH}/model.pkl")
