import numpy as np


class CostFunction:

    @staticmethod
    def cost(Y_pred, Y_expected):
        """
        Find out how wrong we were
        :param Y_pred:
        :param Y_expected:
        :return: The Mean Squared Error
        """
        m = Y_expected.shape[1]
        return (1/m) * np.sum(0.5 * ((Y_expected - Y_pred) ** 2))

    @staticmethod
    def cost_grad(Y_pred, Y_expected):
        """
        :param Y_pred:
        :param Y_expected:
        :return: Gradient of our cost function wrt. Y_pred
        """
        return (Y_expected - Y_pred) * - 1
