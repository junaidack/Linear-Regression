import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
   
    s = sys.argv[1]
    df = pd.read_csv(s)
    df['year'] = df['year'].astype(int)
    df['days'] = df['days'].astype(int)

    plt.plot(df['year'], df['days'])
    plt.ylabel('Number of frozen days')
    plt.xlabel('Year')
    plt.savefig("plot.jpg")

    a = df['year'].to_numpy(dtype=int)
    b = np.ones(len(a), dtype=int)
    c = np.vstack((b,a))
    X = np.transpose(c)

    print("Q3a:")
    print(X)

    Y = df['days'].to_numpy(dtype=int)

    print("Q3b:")
    print(Y)


    Z = np.transpose(X).dot(X)

    print("Q3c:")
    print(Z)

    I = np.linalg.inv(Z)

    print("Q3d:")
    print(I)

    PI = I.dot(np.transpose(X))

    print("Q3e:")
    print(PI)

    hat_beta = PI.dot(Y)

    print("Q3f:")
    print(hat_beta)

    y_hat = hat_beta[0] + hat_beta[1]*2021

    print("Q4: " + str(y_hat))

    sign = np.sign(hat_beta[1])

    if sign == -1:
        print("Q5a: " + str("<"))

    elif sign == 1:
        print("Q5a: " + str(">"))

    else:
        print("Q5a: " + str("="))

    print("Q5b: " + str("The negative sign implies that over time, the amount of ice on Mendota (measured in days the lake is frozen) is decreasing."))


    x_star = (-1*hat_beta[0])/hat_beta[1]

    print("Q6a: " + str(x_star))

    print("Q6b: " + str("This value for x* makes a decent amount of sense if you extrapolate the trend of the ice on Mendota to continue to linearly decrease at approximately the rate it has been"))


main()

