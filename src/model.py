import numpy as np
import copy
from typing import Tuple


class LogisticRegression:
    """A class used to represent a Logistic Regression model"""

    def __init__(
        self,
        C: float = 0.1,
        g_lambda: float = 0.001,
        tol: float = 1e-4,
        max_iter: int = 1000,
    ):
        """
        The constructor for LogisticRegression class.

        Parameters:
            C (float): Regularization parameter.
            g_lambda (float): Learning rate.
            tol (float): Tolerance for stopping criteria.
            max_iter (int): Maximum number of iterations.
        """
        self.C = C
        self.g_lambda = g_lambda
        self.tol = tol
        self.max_iter = max_iter
        self.losses = []
        self.train_accuracies = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        The function to train the model.

        Parameters:
            x (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        epoch_num = 0
        loss_prev = 10**7
        loss = 10**6

        while epoch_num < self.max_iter and np.abs(loss_prev - loss) > self.tol:
            pred = self._predict_proba(x)
            loss_prev = loss
            loss = self._compute_loss(y, pred)
            gradients_w, gradient_b = self._compute_gradients(x, y, pred)
            self._update_model_parameters(gradients_w, gradient_b)
            epoch_num += 1

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        The function to compute the loss.

        Parameters:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: Loss value.
        """
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def _compute_gradients(
        self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        The function to compute the gradients.

        Parameters:
            x (np.ndarray): Training data.
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            Tuple[np.ndarray, float]: Gradients for weights and bias.
        """
        difference = y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(2 * x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])
        return gradients_w, gradient_b

    def _update_model_parameters(self, error_w: np.ndarray, error_b: float) -> None:
        """
        The function to update the model parameters.

        Parameters:
            error_w (np.ndarray): Gradients for weights.
            error_b (float): Gradient for bias.
        """
        self.weights = self.weights - self.g_lambda * error_w
        self.bias = self.bias - self.g_lambda * error_b

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        The function to predict the class labels for the provided data.

        Parameters:
            x (np.ndarray): The test data.

        Returns
                np.ndarray: Predicted class labels for each data point.
        """
        probabilities = self._predict_proba(x)
        return np.where(probabilities > 0.5, 1, 0)

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        The function to compute probabilities for the provided data.

        Parameters:
            x (np.ndarray): The test data.

        Returns:
            np.ndarray: Predicted probabilities for each data point of target=1.
        """
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return probabilities

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        The function to compute probabilities for the provided data.

        Parameters:
            x (np.ndarray): The test data.

        Returns:
            np.ndarray: Predicted probabilities for each data point.
        """
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        y_prob = np.column_stack((1 - probabilities, probabilities))
        return y_prob

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        The function to calculate the sigmoid function.

        Parameters:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Sigmoid values for each data point.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _transform_x(x: np.ndarray) -> np.ndarray:
        """
        The function to transform the input data.

        Parameters:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return copy.deepcopy(x)

    @staticmethod
    def _transform_y(y: np.ndarray) -> np.ndarray:
        """
        The function to transform the target values.

        Parameters:
            y (np.ndarray): Target values.

        Returns:
            np.ndarray: Transformed target values.
        """
        return copy.deepcopy(y)
