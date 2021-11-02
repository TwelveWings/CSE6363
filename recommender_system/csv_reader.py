import csv

from movie import Movie
from rating import Rating

def read_movie_csv(file_name, movie_id = None):
  """
  Reads CSV file with movie data.
  """ 
  i = 0

  csv_file = []

  movie_indices = {}

  genres = set([])

  try:
    with open(file_name, newline='') as csvfile:
      r = csv.reader(csvfile, delimiter=',', quotechar='"')

      for row in r:
        if i == 0:
          i += 1
          continue

        if movie_id is not None and int(row[0]) != movie_id:
          continue

        movie_indices[int(row[0])] = i

        # Create movie class using movieid, title, and genre
        movie = Movie(row[0], row[1], row[2])

        if movie.genre == "(no genres listed)":
          genre = None
          genres.add(genre)
        else:
          movie_genres = movie.genre.split('|')

          for mg in movie_genres:
            genres.add(mg)

        csv_file.append(movie)

        i += 1

    return (csv_file, movie_indices, genres, True)
  except FileNotFoundError:
    print("File does not exist. Matrix factorization process did not run.")

    return (csv_file, movie_indices, genres, False)

def read_rating_csv(file_name, user_id = None):
  """
  Reads CSV file with movie data.
  """
  curr_id = 0
  i = 0
  start_index = 0

  csv_file = []

  user_indices = {}

  try:
    with open(file_name, newline='') as csvfile:
      r = csv.reader(csvfile, delimiter=',', quotechar='"')

      for row in r:
        if i == 0:
          i+= 1
          continue

        if user_id is not None and int(row[0]) != user_id:
          continue

        if int(row[0]) != curr_id:
          curr_id  = int(row[0])
          start_index = i

        user_indices[int(row[0])] = (start_index, i)

        # Create rating class using userid, movieid, rating,
        # and timestamp
        rating = Rating(row[0], row[1], row[2], row[3])

        csv_file.append(rating)

        i += 1

    return (csv_file, user_indices, True)
  except FileNotFoundError:
    print("File does not exist. Matrix factorization process did not run.")

    return (csv_file, user_indices, False)
