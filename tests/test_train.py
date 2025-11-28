from src.train import model, accuracy

def test_model_training():
    # Check model exists
    assert model is not None

    # Accuracy must be > 0
    assert accuracy > 0
