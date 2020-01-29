import numpy as np

def find_nearest_idx(np_arr, value):
  """ Return index in array that is closest to value
  """
  return (np.abs(np_arr - value)).argmin()