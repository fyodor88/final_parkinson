from src.inference.prediction_mapping import summarize_binary_prediction


def test_prediction_confidence_matches_winning_class_in_standard_order():
    summary = summarize_binary_prediction([0, 1], [0.55, 0.45])

    assert summary["label"] == 0
    assert summary["diagnosis"] == "Healthy / Control"
    assert summary["confidence"] == 0.55
    assert summary["class_probabilities"] == {"Healthy": 0.55, "Parkinsons": 0.45}


def test_prediction_mapping_respects_model_class_order():
    summary = summarize_binary_prediction([1, 0], [0.62, 0.38])

    assert summary["label"] == 1
    assert summary["diagnosis"] == "Parkinson's Likely"
    assert summary["confidence"] == 0.62
    assert summary["parkinsons_probability"] == 0.62
    assert summary["class_probabilities"] == {"Parkinsons": 0.62, "Healthy": 0.38}