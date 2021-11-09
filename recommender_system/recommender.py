import numpy as np

def determine_user_preferences(user_id, user_index, movie_indices, ratings, movies, genres):
  """
  Creates a list of genres in movies, sorted by the number of times user_id has watched that
  genre and the average rating per genre.
  """
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
  """
  Creates a list of movies watched by user_id in the top 5 genres.
  """
  preferred_genres = set([])
  movies_watched_in_genres = set([])
  movie_watched_ratings = {}

  i = 0
  for preference in preferences[:4]:
    preferred_genres.add(preference[0])

  try:
    for i in range(user_index[user_id][0] - 1, user_index[user_id][1]):
      index = movie_indices[ratings[i].movie_id]
      for mg in movies[index].genre.split('|'):
        if mg in preferred_genres:
          movies_watched_in_genres.add(ratings[i].movie_id)
          movie_watched_ratings[ratings[i].movie_id] = ratings[i].rating
          break
  except:
    print("get_movies_user_watched_in_genres() : user_id: %s" % str(user_id))

  return (movies_watched_in_genres, movie_watched_ratings)

def find_similar_user_ratings(u_ids, u_id, user_index, ratings, movies_watched):
  """
  Creates a list of 9 users who have watched at least 70% of the same movies as u_id.
  """
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

    try:
      for i in range(user_index[user][0] - 1, user_index[user][1]):
        if ratings[i].movie_id in movies_watched:
          total_movies_watched_by_other += 1
          user_watched.add(ratings[i].movie_id)
          user_ratings[ratings[i].movie_id] = ratings[i].rating
    except:
      print("find_similar_user_ratings() : user: %s" % str(user))

    if float(total_movies_watched_by_other) >= len(movies_watched) * .7:
      recommenders.append((user, user_watched, user_ratings))

  return recommenders

def get_movies_based_on_preference(movies, preferences):
  """
  Creates a list of all movies in movies that are in the top 5 genres preferred by the
  specified user.
  """
  top5_genres = set([preference[0] for preference in preferences])

  movies_in_top5 = set([])

  for movie in movies:
    for mg in movie.genre.split('|'):
      if mg in top5_genres:
        movies_in_top5.add(movie.movie_id)
        break

  return movies_in_top5
