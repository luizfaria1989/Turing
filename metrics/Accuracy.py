import numpy as np

class Accuracy:
    """Calculates the accuracy metric for classification problems.

    Accuracy measures the proportion of correct predictions out of the
    total number of samples. It is calculated as:
    Accuracy = (Correct Predictions) / (Total Predictions)
    """

    def __init__(self):
        """Initializes the accuracy calculator."""
        pass

    def calculate(self, y_pred, y_true):
        """Calculates the accuracy for a batch of predictions.

        Args:
            y_pred (np.ndarray): The predicted outputs from the model, typically
                as probability distributions (output from Softmax).
                Shape: (batch_size, num_classes).
            y_true (np.ndarray): The true labels in one-hot encoded format.
                Shape: (batch_size, num_classes).

        Returns:
            float: The accuracy score, a value between 0 and 1.
        """
        # Convert probabilities to class predictions (the index of the highest value)
        predictions = np.argmax(y_pred, axis=1)

        # Convert one-hot encoded true labels back to class labels
        true_labels = np.argmax(y_true, axis=1)

        # Compare predictions with true labels and sum the matches
        correct = np.sum(predictions == true_labels)

        # Calculate the accuracy
        accuracy = correct / len(predictions)

        return accuracy