# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Generate data for linear regression
slope = 4.5
x = 100*np.random.rand(100,1)
bias = np.ones([100,1])
noise = 20*np.random.randn(100, 1)
X = np.concatenate((bias,x), axis=1)
Y = bias + slope * x + noise

# # # Create a plot to see synthetic data looks like
# fig, ax = plt.subplots(3, 1, figsize=(10, 12))  # Adjust figsize as needed
# # Scatter plots
# ax[0].scatter(x, Y, label='X1 vs Y', color='b')
# ax[1].scatter(bias, Y, label='Bias vs Y', color='g')
# # ax[2].scatter(x3, y, label='X3 vs Y', color='r')
# # ax[0].set_title('X1 vs Y')
# # ax[1].set_title('X2 vs Y')
# # ax[2].set_title('X3 vs Y')
# plt.show()

# Find the optimal model parameters using gradient descent (rather than analytical solutions)
# Initialize model parameters
costs = [] #keep track of squared error cost
theta = np.random.randn(np.size(X,1), 1) / np.sqrt(np.size(X,1))
learning_rate = 0.000001
num_iterations = 10000
m = len(Y)

for iteration in range(num_iterations):
    yhat = np.dot(X, theta)
    error = yhat - Y
    mse = np.dot(error.T,error)/ len(X)
    theta = theta - (learning_rate/m) * np.dot(X.T, error)
    mse = np.sum((error ** 2)) / (2 * m)
    costs.append(mse)

# plot the costs
plt.plot(costs)
# ax.set_title('Cost Function')
plt.show()

# Plot the loss per iteration
plt.plot(yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()






