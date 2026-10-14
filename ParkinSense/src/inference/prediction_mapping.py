"""Helpers for keeping classifier labels and probability outputs aligned."""


def summarize_binary_prediction(classes, probabilities):
    """Return the winning class, its confidence, and named binary probabilities."""
    class_values = [value.item() if hasattr(value, "item") else value for value in classes]
    probability_values = [float(value) for value in probabilities]
    if not class_values or len(class_values) != len(probability_values):
        raise ValueError("Model classes and probabilities must have matching non-empty values.")

    predicted_index = max(range(len(probability_values)), key=probability_values.__getitem__)
    predicted_class = class_values[predicted_index]
    names = {}
    for value in class_values:
        if value in (0, 0.0, "0"):
            names[value] = "Healthy"
        elif value in (1, 1.0, "1"):
            names[value] = "Parkinsons"
        else:
            names[value] = str(value)

    return {
        "label": predicted_class,
        "diagnosis": "Parkinson's Likely" if names[predicted_class] == "Parkinsons" else (
            "Healthy / Control" if names[predicted_class] == "Healthy" else f"Class {predicted_class}"
        ),
        "confidence": probability_values[predicted_index],
        "parkinsons_probability": next(
            (probability_values[index] for index, value in enumerate(class_values)
             if names[value] == "Parkinsons"),
            None,
        ),
        "class_probabilities": {
            names[value]: probability_values[index]
            for index, value in enumerate(class_values)
        },
    }