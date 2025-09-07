import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Paths
DATA_PATH, RESULTS_PATH= "data", "results"

# Import Pandas DataFrame
df = pd.read_csv(f"{DATA_PATH}/data.csv")

# Create Y output
le = LabelEncoder()
y = le.fit_transform(df["category"])

# Create X input
X = df.drop(columns=["category", "invoice_id", "date", "vat_percent", "currency", "paid"])

# Train-test split, DONT maintain the classes percentage distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Classifying the columns for Sklearn Pipeline and calling the Model
model = Model(
    numeric_features=["net_amount"],
    categorical_features=["vendor", "payment_method"],
    text_features="description"
)

# Fit the model
model.fit(X_train, y_train)

# Make a predict
y_pred = model.predict(X_test)
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)


# Get classification report
report = classification_report(y_test_labels, y_pred_labels, labels=le.classes_)
print(f"Classification Report\n{report}")

# Draw confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig(f"{RESULTS_PATH}/confusion_matrix.jpg", format="jpg")

# Save model
with open(f"{RESULTS_PATH}/model.pkl", 'wb') as f:
        pickle.dump(model, f)
