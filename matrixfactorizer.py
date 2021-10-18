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
    self.items = np.random.normal(scale=(1/num_features), size=(len(matrix[0]), num_features))

    self.training_set = []

    for u in range(len(self.users)):
      for i in range(len(self.items)):
        if(matrix[u][i] == 0):
          continue
        self.training_set.append((u, i, self.matrix[u][i]))

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
        predicted_matrix[u][i] = np.dot(self.users[u].T, self.items[i])

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

    users = np.copy(self.users)
    items = np.copy(self.items)

    for element in training:
      user, item, actual = element

      # Determine the user rating by taking the dot product of user 
      # and item matrices.
      user_rating = np.dot(self.users[user].T, self.items[item])

      error = actual - user_rating

      # Calculate sum of squared error
      SSE += (error * error)

      # Copy users matrix so that updates values do not affect update to items matrix.
      uc = np.copy(users)

      # Update the user and item matrices for each feature.
      for k in range(self.num_features):
        users[user][k] += self.learning_rate * (error * items[item][k] - self.reg * users[user][k])
        items[item][k] += self.learning_rate * (error * uc[user][k] - self.reg * items[item][k])

    return (SSE, users, items)

  def SSE(self, errors):
    """
    Finds the sum of squares error.
    """
    sum_of_squared_error = 0.0

    for error in errors:
      sum_of_squared_error += (error * error)

    return sum_of_squared_error

  def train(self):
    """
    Trains the model based on algorithm in paper by Takács et. al.
    """
    training_set = self.training_set

    # Randomize data set
    np.random.shuffle(training_set)

    training_index = int(math.floor(len(training_set) * 0.8))

    # Split data set into training set and validation set
    training, validation = training_set[:training_index], training_set[training_index:]

    current_RMSE = float('inf')
    iterations_of_no_decrease = 0
    iterations = 0
    min_loss = float('inf')

    # Loop until RMSE has not decreased for two iterations.
    while True:
      if iterations_of_no_decrease > 1 or iterations == self.max_iterations:
        break

      error, users, items = self.SGD(training)

      root_mean_squared_error = self.RMSE(error, len(validation))

      # If the current RMSE calculated is less than the minimum loss, 
      # update self.users and self.items
      if root_mean_squared_error < min_loss:
        min_loss = root_mean_squared_error
        self.users = users
        self.items = items

      if root_mean_squared_error >= current_RMSE:
        iterations_of_no_decrease += 1
      else:
        iterations_of_no_decrease = 0

      current_RMSE = root_mean_squared_error

      iterations += 1
