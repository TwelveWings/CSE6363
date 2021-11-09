import numpy as np
import re
import sys

from csv_reader import read_movie_csv, read_rating_csv
from matrixfactorizer import MatrixFactorizer
from movie import Movie
from recommender import determine_user_preferences, get_movies_user_watched_in_genres
from recommender import find_similar_user_ratings, get_movies_based_on_preference

def generate_matrix(u_index, u_id, recommenders, movies_in_top5, movies_watched, ratings):
  """
  Generates a matrix from a the number of users and movies in top 5 genre.
  """
  matrix = []
  matrix_row = []

  list_of_movies = list(movies_in_top5)

  movies_not_watched = {}
  movies_not_watched_index = []

  list_of_movies.sort()
  
  for i in range(10):
    index = 0
    matrix_row = []

    for movie in list_of_movies:
      if i == 0:
        try:
          matrix_row.append(movies_watched[movie])
        except:
          matrix_row.append(0)
          movies_not_watched[index] = movie
          movies_not_watched_index.append(index)
      else:
        try:
          matrix_row.append(recommenders[i - 1][2][movie])
        except:
          matrix_row.append(0)
      
      index += 1

    matrix.append(matrix_row)

  matrix = np.array(matrix)

  return (matrix, movies_not_watched, movies_not_watched_index)

if __name__ == '__main__':
  print("Loading movies...")
  movies, movie_indices, genres, movie_file_exists = read_movie_csv("movies.csv")
  print("Loading users...")
  ratings, user_indices, rating_file_exists = read_rating_csv("ratings.csv")

  user_ids = [i for i in range(1, ratings[-1].user_id)]

  while(True):
    print("\nPlease enter user ID or enter '-1' to quit: ")

    try:
      user_id = int(input())
    except:
      print("Invalid user ID. Please enter a positive integer.\n")
      continue

    if user_id == -1:
      print("Program terminated.")
      break
    elif user_id < -1 or user_id > len(user_ids):
      print("Invalid ID. Please try again.")
      continue

    print("Loading preferences...\n")

    # Determine which genres the user most watches and their respective rating.
    # Sort each genre by the number of items watched and the average rating for
    # the genre.
    preferences = determine_user_preferences(
      user_id, user_indices, movie_indices, ratings, movies, genres)

    # Get top 5 based on items watched and average rating.
    top5Genres = preferences[:5]

    # Get list of all movies watched by the user in the top 5 genres.
    movies_watched_in_genres, movie_watched_ratings = get_movies_user_watched_in_genres(
      user_id, user_indices, movie_indices, ratings, movies, top5Genres)

    print("Top 5 genres for ID %s" % str(user_id))

    i = 0
    for preference in top5Genres:
      print("%s) %s" % (str(i + 1), str(preference[0])))
      i += 1

    # Get list of all movies in dataset that are in the genres preferred by the user.
    movies_in_top5 = get_movies_based_on_preference(movies, top5Genres)

    # Determine users who have watched the same movies as the specified user.
    recommenders = find_similar_user_ratings(
      user_ids, user_id, user_indices, ratings, movies_watched_in_genres)

    print("Generating prediction matrix...\n")
    matrix, movies_not_watched, not_watched_index = generate_matrix(
      user_indices, user_id, recommenders, movies_in_top5, movie_watched_ratings, 
      ratings)

    print("Matrix factorization...\n")
    mf = MatrixFactorizer(matrix, 0.001, 0.001, 2, 2500)
    mf.train()

    prediction = mf.get_predicted_matrix()
    user_prediction = list(prediction[0])

    recommendations = 0

    # Determine the highest prediction based on the matrix factorization. If the movie has
    # not been watched, recommend it. Otherwise, move to the next highest.
    print("Recommendations:")
    while(recommendations < 5):
      movie_recommendation = user_prediction.index(max(user_prediction))

      if movie_recommendation in not_watched_index:
        movie_id = movies_not_watched[movie_recommendation]
        movie_index = movie_indices[movie_id]
        movie_title = movies[movie_index].title
        movie_genres = movies[movie_index].genre
        print("%s) %s %s" % (str(recommendations + 1), movie_title, movie_genres))
        recommendations += 1

      user_prediction[movie_recommendation] = 0.0

