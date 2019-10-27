def cost(Y_pred, Y_expected):
    """
    Find out how wrong we were
    :param Y_pred:
    :param Y_expected:
    :return: The Mean Squared Error
    """
    return 0.5 * ((Y_expected - Y_pred) ** 2)


def cost_grad(Y_pred, Y_expected):
    """
    :param Y_pred:
    :param Y_expected:
    :return: Gradient of our cost function wrt. Y_pred
    """
    return Y_expected - Y_pred
