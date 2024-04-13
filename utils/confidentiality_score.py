import math


def compute_confidence_scores(distances: list) -> list:
    """
    Compute normalized confidence scores from a list of distances using exponential decay.

    Args:
    distances (list of floats): A list of distances, where smaller distances indicate higher similarity.

    Returns:
    list of str: Normalized confidence scores as percentages.
    """
    # Calculate confidence scores using exponential decay
    confidence_scores = [math.exp(-distance) for distance in distances]

    # Normalize confidence scores
    total_score = sum(confidence_scores)
    normalized_scores = [
        f"{score * 100.0 / total_score:.2f}%" for score in confidence_scores]

    return normalized_scores
