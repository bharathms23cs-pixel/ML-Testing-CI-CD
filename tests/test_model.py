import joblib
import pandas as pd


def test_model_exists():
    model = joblib.load("model (1).pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("model (1).pkl")

    data = pd.read_csv("house (2).csv")
    X = data.drop("MedHouseVal", axis=1)

    prediction = model.predict(X.iloc[[0]])

    assert len(prediction) == 1


def test_prediction_is_positive():
    model = joblib.load("model (1).pkl")

    data = pd.read_csv("house (2).csv")
    X = data.drop("MedHouseVal", axis=1)

    prediction = model.predict(X.iloc[[0]])

    assert prediction[0] >= 0
