from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Model():
    def __init__(self, numeric_features, categorical_features, text_features):
        # Transformers
        self.categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        self.text_transformer = CountVectorizer()
        self.scaler = StandardScaler()

        # Creating Feature Engineering pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.scaler, numeric_features),
                ('cat', self.categorical_transformer, categorical_features),
                ('text', self.text_transformer, text_features)
            ]
        )

        # Creating Classifier pipeline
        self.classifier = Pipeline(
            steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(max_iter=10000, random_state=42))
            ]
        )

    def fit(self, x_train, y_train):
        return self.classifier.fit(x_train, y_train)

    def predict(self, x):
        return self.classifier.predict(x)

    def __str__(self):
        return f"Model: {self.classifier}"
