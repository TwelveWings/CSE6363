import csv
import numpy as np
import re
import sys

from matrixfactorizer import MatrixFactorizer

def read_csv(file_name):
  """
  Reads a given CSV file.
  """
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
  """
  Generates a matrix from a given CSV file.
  """
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

if __name__ == '__main__':
  # If a file with a csv extension is not specified matrix factorization cannot begin.
  if len(sys.argv) > 1 and not re.match('[A-Za-z0-9_]+[.][c][s][v]', sys.argv[1]):
    print("Invalid CSV file specified. Please specify a valid CSV file.")
  elif len(sys.argv) > 1:
    file_name = sys.argv[1]
    num_features = 2
    max_iteration = 2500
    learning_rate = 0.001

    # If a second argument is provided to the execution command, it must be an integer value specifying the number of iterations.
    # Any iterations less than 1 are invalid.
    if len(sys.argv) > 2:
      try:
        if(int(sys.argv[2]) < 1):
          raise ValueError

        max_iteration = int(sys.argv[2])
      except:
        print("Invalid number of iterations specified. Iterations must be a positive integer. Process has defaulted to 2500 iterations.")

    csv_file, file_exists = read_csv(file_name)

    print(max_iteration)

    if file_exists:
      matrix = generate_matrix(csv_file)

      mf = MatrixFactorizer(matrix, learning_rate, 0.001, num_features, max_iteration)
      mf.train()
      mf.display_matrices()
