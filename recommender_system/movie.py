class Movie():
  """
  A movie from movies.CSV
  """
  def __init__(self, movie_id, title, genre):
    self.movie_id = int(movie_id)
    self.title = title
    self.genre = genre
