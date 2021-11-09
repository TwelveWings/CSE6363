class Rating():
  """
  A rating from ratings.CSV
  """
  def __init__(self, user_id, movie_id, rating, timestamp):
    self.user_id = int(user_id)
    self.movie_id = int(movie_id)
    self.rating = float(rating)
    self.timestamp = timestamp
