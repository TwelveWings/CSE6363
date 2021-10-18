This program was developed to use matrix factorization using stochastic gradient descent. The program creates a data set based on the input matrix containing 3-tuples consisting of a user, an item, and the rating in the input matrix: (u, i, r) where u is an integer within the range of [0, number of users], i is an integer within the range of [0, number of items], and r is an integer in the range of [1, 10]. Any zero integer ratings in input matrix are ignored as those are treated as "missing" values.

Once the data set is created, it is shuffled and split into a training and validation set. A user matrix of shape (n, k) where n is the number of users and k is the number of features and an item matrix of shape (k, m) where m is the number of items are created containing randomly generated values. From there, the main algorithm begins.

1) First, the stochastic gradient descent (SGD) process begins by determining a predicted user rating by finding the dot product of the user's feature values and the items feature values. 

2) This predicted rating is subtracted from the actual rating to create an error

3) The error is then saved in a list. 

4) Once the error is computed, the user and item matrices are updated for each feature.

5) Once the SGD it run for the current iteration, the sum of squares error (SSE) and root mean squared error (RMSE) are calculated using the formula in paper by [1].

 SSE = sum(error * error)

 RMSE = sqrt(SSE / |validation set|)

 SSE is just the sum of all the squared errors; RMSE is the square root of the SSE divided by the cardinality of the validation set.

6) The process repeats until the RMSE has not decreased for two iterations or 5000 iterations have passed.

This program is written in Python 3. To run it use the command:

python3 main.py <csv_file>.csv

Depending on your environment, you may be able to run it using:

python main.py <csv_file>.csv

<csv_file> must be a valid file name and it must have the extension .csv; otherwise, the program will not run. 

Once the program is finished executing, it will display the final user matrix, item matrix, the predicted matrix and the actual (input) matrix.
