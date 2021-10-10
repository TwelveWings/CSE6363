import csv
import math
import numpy as np
import re
import sys

def read_csv(file_name):
  
  csv_data = []

  try:
    with open(file_name, newline='') as csvfile:
      r = csv.reader(csvfile, delimiter=' ', quotechar='|')

      for row in r:
        csv_data.append(row)

    return (csv_data, True)
  except FileNotFoundError:
    print("File does not exist. Matrix factorization process did not run.")

    return (csv_data, False)

def generate_matrix(csv_data):
  matrix = []
  matrixRow = []

  for row in csv_file:
    for integer in row[0].split(','):
      if len(integer.strip()) == 0:
        matrixRow.append(0)
      else:
        matrixRow.append(int(integer))

    matrix.append(matrixRow)
    matrixRow = []

  matrix = np.array(matrix)

  return matrix


class MatrixFactorizer():
  def __init__(self, matrix, learning_rate, regularization, num_features):
    self.learning_rate = learning_rate
    self.reg = regularization
    self.num_features = num_features

    np.random.seed(1)

    self.matrix = matrix
    self.users = np.random.normal(scale=1/num_features, size=(len(matrix), num_features))
    self.items = np.random.normal(scale=1/num_features, size=(num_features, len(matrix[0])))

    self.training_set = []

    for u in range(len(self.users)):
      for i in range(len(self.items.T)):
        self.training_set.append((self.users[u], self.items.T[i], self.matrix[u][i]))

  def display_matrices(self):
    """
    Displays a user matrix, item matrix, input matrix and prediction matrix.
    """
    # Print input matrix, factored user matrix, and factored item matrix    
    print("\nITEM MATRIX:\n")
    print(self.items.T.shape)
    print(self.items.T)

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

  def get_error(self, actual_score, estimation):
    return actual_score - estimation

  def get_predicted_matrix(self):
    """
    Based on the user and item matrices, creates a prediction matrix.
    """
    predicted_matrix = np.zeros((len(self.users), len(self.items.T)))

    print(predicted_matrix.shape)

    for u in range(len(self.users)):
      for i in range(len(self.items.T)):
        predicted_matrix[u][i] = np.dot(self.users[u].T, self.items.T[i])

    return predicted_matrix

  def RMSE(self, sum_of_squared_error, cardinality):
    return np.sqrt(sum_of_squared_error / cardinality)

  def SGD(self, training):
    errors = []

    for element in training:
      user, item, actual = element

      user_rating = np.dot(user.T, item)

      error = self.get_error(actual, user_rating)

      errors.append(error)

      for u in range(len(self.users)):
        for i in range(len(self.items)):
          for k in range(self.num_features):
            uc = np.copy(self.users)
            self.users[u][k] += self.learning_rate * (error * self.items[k][i] - self.reg * self.users[u][k])
            self.items[k][i] += self.learning_rate * (error * uc[u][k] - self.reg * self.items[k][i])

    return errors

  def SSE(self, errors):
    sum_of_squared_error = 0.0

    for error in errors:
      sum_of_squared_error += (error * error)

    return sum_of_squared_error

  def train(self):
    training_set = self.training_set
    users = self.users
    items = self.items

    np.random.shuffle(training_set)

    training_index = int(math.floor(len(training_set) * 0.7))

    user_out = np.copy(self.users)
    item_out = np.copy(self.items)

    training, validation = training_set[:training_index], training_set[training_index:]

    current_RMSE = float('inf')
    iterations_of_no_decrease = 0
    loss = []
    it = 0

    while True:
      if iterations_of_no_decrease > 1:
        break

      errors = self.SGD(training)

      sum_of_squared_error = self.SSE(errors)
      root_mean_squared_error = self.RMSE(sum_of_squared_error, len(validation))

      if len(loss) > 0 and root_mean_squared_error < min(loss):
        user_out = np.copy(users)
        item_out = np.copy(items)

      if root_mean_squared_error >= current_RMSE:
        iterations_of_no_decrease = 0
      else:
        iterations_of_no_decrease += 1

    self.users = user_out
    self.items = item_out

if __name__ == '__main__':
  if len(sys.argv) > 1 and not re.match('[A-Za-z0-9_]+[.][c][s][v]', sys.argv[1]):
    print("Invalid CSV file specified. Please specify a valid CSV file.")
  elif len(sys.argv) > 1:
    file_name = sys.argv[1]
    num_features = 2
    learning_rate = 0.01

    csv_file, file_exists = read_csv(file_name)

    if file_exists:
      matrix = generate_matrix(csv_file)

      mf = MatrixFactorizer(matrix, learning_rate, 1, num_features)
      mf.train()
      mf.display_matrices()
