import numpy as np
import re
import sqlite3
import sys

from matrixfactorizer import MatrixFactorizer
from movie import Movie
from csv_reader import read_movie_csv, read_rating_csv

def determine_user_preferences(user_id, user_index, movie_indices, ratings, movies, genres):
  start, end = user_index[user_id]

  user_ratings = ratings[start - 1 : end]

  genre_avg_rating = {}
  times_genre_watched = {}
  movies_watched = set([])

  for genre in genres:
    genre_avg_rating[genre] = 0.0
    times_genre_watched[genre] = 0
 
  for user_rating in user_ratings:
    index = movie_indices[user_rating.movie_id]
    movies_watched.add(user_rating.movie_id)
    movie_genres = movies[index].genre.split('|')

    for mg in movie_genres:
      genre_avg_rating[mg] += user_rating.rating
      times_genre_watched[mg] += 1

  genre_preferences = []
 
  for genre in times_genre_watched:
    if times_genre_watched[genre] > 0:
      genre_avg_rating[genre] /= times_genre_watched[genre]

    genre_preferences.append((genre, times_genre_watched[genre], genre_avg_rating[genre]))

  genre_preferences.sort(key=lambda tup: (tup[1], tup[2]), reverse=True)

  return genre_preferences

def get_movies_user_watched_in_genres(user_id, user_index, movie_indices, ratings, movies, preferences):
  preferred_genres = set([])
  movies_watched_in_genres = set([])
  movie_watched_ratings = {}

  i = 0
  for preference in preferences[:4]:
    preferred_genres.add(preference[0])

  for i in range(user_index[user_id][0], user_index[user_id][1] + 1):
    index = movie_indices[ratings[i].movie_id]
    for mg in movies[index].genre.split('|'):
      if mg in preferred_genres:
        movies_watched_in_genres.add(ratings[i].movie_id)
        movie_watched_ratings[ratings[i].movie_id] = ratings[i].rating
        break

  return (movies_watched_in_genres, movie_watched_ratings)

def find_similar_user_ratings(u_ids, u_id, user_index, ratings, movies_watched):
  other_users = u_ids.copy()

  other_users.remove(u_id)

  np.random.shuffle(other_users)

  recommenders = []

  for user in other_users:
    user_watched = set([])
    user_ratings = {}
    if len(recommenders) > 8:
      break

    total_movies_watched_by_other = 0
    total_similar_ratings = 0

    for i in range(user_index[user][0], user_index[user][1] + 1):
      if ratings[i].movie_id in movies_watched:
        total_movies_watched_by_other += 1
        user_watched.add(ratings[i].movie_id)
        user_ratings[ratings[i].movie_id] = ratings[i].rating

    if float(total_movies_watched_by_other) >= len(movies_watched) * .7:
      recommenders.append((user, user_watched, user_ratings))

  return recommenders

def get_movies_based_on_preference(movies, preferences):
  top5 = preferences[:4]

  top5_genres = set([preference[0] for preference in top5])

  movies_in_top5 = set([])

  for movie in movies:
    for mg in movie.genre.split('|'):
      if mg in top5_genres:
        movies_in_top5.add(movie.movie_id)
        break

  return movies_in_top5

def generate_matrix(u_index, u_id, recommenders, preferences, movies_in_top5, movies_watched, ratings):
  """
  Generates a matrix from a given CSV file.
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

  """
  for movie in list_of_movies:
    print(movie)
    print(movies_watched)
    try:
      matrix_row.append(movies_watched[movie])
    except:
      movies_not_watched.add(movie)

  matrix.append(matrix_row)

  for i in range(10):
    matrix_row = []
    if i == 9:
      break
    for j in range(len(movies_in_top5)):
      try:
        matrix_row.append(recommenders[i][2][list_of_movies[j]])
      except:
        matrix_row.append(0)

    matrix.append(matrix_row)

  matrix = np.array(matrix)
  """
  return (matrix, movies_not_watched, movies_not_watched_index)

if __name__ == '__main__':
  print("Loading movies...")
  movies, movie_indices, genres, movie_file_exists = read_movie_csv("movies.csv")
  print("Loading users...")
  ratings, user_indices, rating_file_exists = read_rating_csv("ratings.csv")

  user_ids = [i for i in range(ratings[-1].user_id)]

  while(True):
    print("Please enter user ID or enter '-1' to quit: ")

    try:
      user_id = int(input())
    except:
      print("Invalid user ID. Please enter a positive integer.\n")
      continue

    if user_id == -1:
      print("Program terminated.")
      break

    print("Loading preferences...\n")

    preferences = determine_user_preferences(
      user_id, user_indices, movie_indices, ratings, movies, genres)

    movies_watched_in_genres, movie_watched_ratings = get_movies_user_watched_in_genres(
      user_id, user_indices, movie_indices, ratings, movies, preferences)

    print("Top 5 genres for ID %s" % str(user_id))

    i = 0
    for preference in preferences:
      if i > 4:
        break

      print("%s) %s" % (str(i + 1), str(preference[0])))
      i += 1

    movies_in_top5 = get_movies_based_on_preference(movies, preferences)

    recommenders = find_similar_user_ratings(
      user_ids, user_id, user_indices, ratings, movies_watched_in_genres)

    print("Generating prediction matrix...\n")
    matrix, movies_not_watched, not_watched_index = generate_matrix(
      user_indices, user_id, recommenders, preferences, movies_in_top5, 
      movie_watched_ratings, ratings)

    print("Matrix factorization...\n")
    mf = MatrixFactorizer(matrix, 0.001, 0.001, 2, 2500)
    mf.train()

    prediction = mf.get_predicted_matrix()
    user_prediction = list(prediction[0])

    recommendations = 0

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
      del user_prediction[movie_recommendation]
