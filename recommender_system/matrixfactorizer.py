import math
import numpy as np

class MatrixFactorizer():
  """
  Class for performing matrix factorization.
  """
  def __init__(self, matrix, learning_rate, regularization, num_features, iterations):
    self.learning_rate = learning_rate
    self.reg = regularization
    self.num_features = num_features
    self.max_iterations = iterations

    np.random.seed(1)

    self.matrix = matrix

    self.users = np.random.normal(scale=(1/num_features), size=(len(matrix), num_features))
    self.u_bias = np.random.randn(len(matrix))
    self.items = np.random.normal(scale=(1/num_features), size=(len(matrix[0]), num_features))
    self.i_bias = np.random.randn(len(matrix[0]))

    self.training_set = []
    self.ratings = []

    for u in range(len(self.users)):
      for i in range(len(self.items)):
        if(matrix[u][i] == 0):
          continue
        self.training_set.append((u, i, self.matrix[u][i]))
        self.ratings.append(matrix[u][i])

    self.avg_rating = sum(self.ratings) / len(self.ratings)

  def display_matrices(self):
    """
    Displays a user matrix, item matrix, input matrix and prediction matrix.
    """
    # Print input matrix, factored user matrix, and factored item matrix    
    print("\nITEM MATRIX:\n")
    print(self.items.shape)
    print(self.items)

    print("\nUSER MATRIX:\n")
    print(self.users.shape)
    print(self.users)

    predicted_matrix = self.get_predicted_matrix()
    print("\nPREDICTED MATRIX:\n")
    print(predicted_matrix.shape)
    print(predicted_matrix)

    print("\nINPUT MATRIX:\n")
    print(self.matrix.shape)
    print(self.matrix)

  def get_predicted_matrix(self):
    """
    Based on the user and item matrices, creates a prediction matrix.
    """
    predicted_matrix = np.zeros((len(self.users), len(self.items)))

    for u in range(len(self.users)):
      for i in range(len(self.items)):
        bias = self.avg_rating + self.u_bias[u] + self.i_bias[i]
        predicted_matrix[u][i] = np.dot(self.users[u].T, self.items[i]) + bias
        
    return predicted_matrix

  def RMSE(self, sum_of_squared_error, cardinality):
    """
    Finds the root mean squared error
    """
    return np.sqrt(sum_of_squared_error / cardinality)

  def SGD(self, training):
    """
    Runs stochastic gradient descent
    """
    SSE = 0.0

    for element in training:
      user, item, actual = element

      # Determine the user rating by taking the dot product of user 
      # and item matrices.
      bias = self.avg_rating + self.u_bias[user] + self.i_bias[item]
      user_rating = np.dot(self.users[user].T, self.items[item]) + bias

      error = actual - user_rating

      # Calculated squared error and add it to current sum of squared errors
      SSE += (error * error)

      self.u_bias[user] += self.learning_rate * (error - self.reg * self.u_bias[user])
      self.i_bias[item] += self.learning_rate * (error - self.reg * self.i_bias[item])

      # Copy users matrix so that updates values do not affect update to items matrix.
      uc = np.copy(self.users)

      # Update the user and item matrices for each feature.
      for k in range(self.num_features):
        self.users[user][k] += self.learning_rate * (error * self.items[item][k] - self.reg * self.users[user][k])
        self.items[item][k] += self.learning_rate * (error * uc[user][k] - self.reg * self.items[item][k])

    return SSE

  def train(self):
    """
    Trains the model based on algorithm in paper by Takács et. al.
    """
    training_set = self.training_set

    # Randomize data set
    np.random.shuffle(training_set)

    current_RMSE = float('inf')
    iterations_of_no_decrease = 0
    iterations = 0
    min_loss = float('inf')

    # Loop until RMSE has not decreased for two iterations or 5000 iterations have passed.
    while True:
      if iterations_of_no_decrease > 1 or iterations == self.max_iterations:
        break

      error = self.SGD(training_set)

      root_mean_squared_error = self.RMSE(error, len(training_set))

      if root_mean_squared_error >= current_RMSE:
        iterations_of_no_decrease += 1
      else:
        iterations_of_no_decrease = 0

      current_RMSE = root_mean_squared_error

      iterations += 1