This program is written in Python 3. To run it use the command:

python3 main.py <csv_file>.csv <num_iterations>

Depending on your environment, you may be able to run it using:

python main.py <csv_file>.csv <num_iterations>

<csv_file> must be a valid file name and it must have the extension .csv; otherwise, the program will not run. 

<num_iterations> is an optional argument that must be a positive integer. Not entering this argument or entering an invalid argument will result in 2500 iterations being run by default. 2000-2500 iterations are the minimum recommended number of iterations. Any more iterations will yield more accurate results, but, depending on your machine and the size of the input matrix in the CSV, this process might take several minutes to complete; less iterations may result in less accurate results but shorter processing time.

Once the program is finished executing, it will display the final user matrix, item matrix, the predicted matrix and the actual (input) matrix.
