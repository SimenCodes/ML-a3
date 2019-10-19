def cost(Y_pred, Y_expected):
    """
    Find out how wrong we were
    :param Y_pred:
    :param Y_expected:
    :return: The Mean Squared Error
    """
    return 0.5 * ((Y_pred - Y_expected) ** 2)

def cost_grad():
    pass