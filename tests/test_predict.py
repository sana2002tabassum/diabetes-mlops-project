import os

def test_model_file():
    # Check model file saved
    assert os.path.exists("artifacts/diabetes_model.pkl")
