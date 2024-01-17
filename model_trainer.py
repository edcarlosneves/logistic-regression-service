import uuid
from datetime import datetime

import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = "data"
CUSTOMERS_FILES_DIR = f"{BASE_DIR}/customers_files"
CLASSIFIERS_DIR = f"{BASE_DIR}/classifiers"


def get_timenow_str():
    return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:-5]


class PredictResults:
    def _init__(self, classifier_name):
        self.classifier_name = f"{classifier_name}_classifier.joblib"
        self.scaller_name = f"{classifier_name}_scaller.joblib"
        self.loaded_classifier = load(f"{CLASSIFIERS_DIR}/{self.classifier_name}")
        self.loaded_scaller = load(f"{CLASSIFIERS_DIR}/{self.scaller_name}")

    def predict_result(self, input_to_predict):
        result = self.loaded_classifier.predict(
            self.loaded_scaller.transform([input_to_predict])
        )
        return result


class ModelTrainer:
    def __init__(self, customer_file_name, test_size):
        self.customer_file_name = customer_file_name
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression()

    def prepare_data(self):
        self.dataset = pd.read_csv(f"{CUSTOMERS_FILES_DIR}/{self.customer_file_name}")
        self.x = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values

    def split_train_and_test_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size
        )

    def fit_transform_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.fit_transform(self.X_test)

    def make_predictions(self):
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)

    def calculate_accuracy(self):
        self.classifier_accuracy = accuracy_score(self.y_test, self.y_pred) * 100

    def make_prefix_name_for_classifier(self):
        classifier_accuracy_str = str(self.classifier_accuracy).replace(".", "_")
        timestr = get_timenow_str()
        self.classifier_prefix_name = str(uuid.uuid4())

    def make_classifier_path_to_save(self):
        self.classifier_path = (
            f"{CLASSIFIERS_DIR}/{self.classifier_prefix_name}_classifier.joblib"
        )

    def make_scaller_path_to_save(self):
        self.scaller_path = (
            f"{CLASSIFIERS_DIR}/{self.classifier_prefix_name}_scaller.joblib"
        )

    def save_classifier(self):
        dump(self.classifier, self.classifier_path)

    def save_scaller(self):
        dump(self.scaler, self.scaller_path)

    def __call__(self):
        self.prepare_data()
        self.split_train_and_test_data()
        self.fit_transform_data()
        self.make_predictions()
        self.calculate_accuracy()
        self.make_prefix_name_for_classifier()
        self.make_classifier_path_to_save()
        self.make_scaller_path_to_save()
        self.save_classifier()
        self.save_scaller()

        return self.classifier_accuracy, self.classifier_prefix_name


model_trainer = ModelTrainer("social_network_ads.csv", 0.75)
print(model_trainer())
