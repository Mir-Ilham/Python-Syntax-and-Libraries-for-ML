import numpy as np

def gradient_descent(exp, test, interview, price):
    exp_curr = bedroom_curr = age_curr = b_curr = 0
    rate = 0.005
    n = len(exp)
    for i in range(100000):
        price_predicted = exp_curr * exp + bedroom_curr * test + age_curr * interview + b_curr
        cost = (1/n) * sum([val**2 for val in (price - price_predicted)])
        exp_d = -(2/n)*sum(exp * (price - price_predicted))
        bedroom_d = -(2/n)*sum(test * (price - price_predicted))
        age_d = -(2/n)*sum(interview * (price - price_predicted))
        yd = -(2/n)*sum(price - price_predicted)
        exp_curr = exp_curr - rate * exp_d
        bedroom_curr = bedroom_curr - rate * bedroom_d
        age_curr = age_curr - rate * age_d
        b_curr = b_curr - rate * yd
        print(f"exp {exp_curr}, test {bedroom_curr}, interview {age_curr}, b {b_curr}, cost {cost}, iteration {i}")

def gradient_descent(features, output):
    n = len(features)
    coefficients = np.array([0] * n)
    intercept = 0
    rate = 0.001

    for i in range(100000):
        output_predicted = intercept
        for i in range(n):
            output_predicted += (coefficients[i] * features[i])
        # cost = (1/n) * sum([val**2 for val in (output - output_predicted)])
        for i in range(n):
            feature_derivative = -(2/n)*sum(features[i] * (output - output_predicted))
            coefficients[i] = coefficients[i] - rate * feature_derivative
        intercept_derivative = -(2/n)*sum(output - output_predicted)
        intercept = intercept - rate * intercept_derivative

    print(coefficients)
    print(intercept)
    # print(cost)

exp = np.array([0, 0, 5, 2, 7, 3, 10, 11])
test = np.array([8, 8, 6, 10, 9, 7, 7, 7])
interview = np.array([9, 6, 7, 10, 6, 10, 7, 8])
price = np.array([50000, 45000, 60000, 65000, 70000, 62000, 72000, 80000])

features = np.array([exp, test, interview])
gradient_descent(features, price)
# gradient_descent(exp, test, interview, price)