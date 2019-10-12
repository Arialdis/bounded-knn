import sys, time, math
import pandas as pd
import numpy as np
from operator import itemgetter

class Vector:
  def __init__(self, id, label, vector, angle):
    self.id = id
    self.label = label
    self.vector = vector
    self.magnitude = math.sqrt(sum((vector)**2))
    self.angle = angle

class Neighbor:
  def __init__(self, id, label, distance):
    self.id = id
    self.label = label
    self.distance = distance

class Model:
  def __init__(self, vector_model, neighbor_model):
    self.vector_model = vector_model
    self.neighbor_model = neighbor_model

class BKNN:
  def execute(self, train_data, test_data, k):
    # create mean and max origins
    train_matrix = train_data.iloc[:, 2:].values
    mean_origin = train_matrix.mean(axis = 0)
    max_origin = train_matrix.max(axis = 0)

    # create origin vector
    if np.array_equal(mean_origin, max_origin):
      print("Error: unable to create origin vector. The mean and max origins are the same.")
      sys.exit(1)
    
    origin_vector = Vector(None, None, np.subtract(max_origin, mean_origin), None)

    # create model
    model = self.generate_model(train_data, mean_origin, origin_vector)

    # track correct predictions
    correct = 0

    # classify test data
    for i in range(len(test_data)):
      # create vector from mean_origin to test sample
      test_vector = self.generate_vector(mean_origin, test_data.iloc[i], origin_vector)

      nearest_neighbors = self.find_nearest_neighbors(test_vector, model, k)
      prediction = self.make_prediction(nearest_neighbors)
      
      if prediction == test_vector.label:
        correct += 1

    # calculate and display accuracy
    print("Accuracy: " + str(correct) + " out of " + str(len(test_data)) + " (" + str(round((float(correct) / len(test_data)) * 100, 2)) + "%)")
  
  def generate_model(self, train_data, mean_origin, origin_vector):
    vector_model = []
    neighbor_model = []

    for i in range(len(train_data)):
      # create vector from mean_origin to train sample
      train_vector = self.generate_vector(mean_origin, train_data.iloc[i], origin_vector)

      # insert train_vector into model in ascending order
      index = self.find_vector_index(train_vector, vector_model, 0, len(vector_model) - 1)
      vector_model.insert(index, train_vector)

      # insert neighbor into model in ascending order
      neighbor = Neighbor(train_vector.id, train_vector.label, train_vector.magnitude)
      index = self.find_neighbor_index(neighbor, neighbor_model, 0, len(neighbor_model) - 1)
      neighbor_model.insert(index, neighbor)
    
    return Model(vector_model, neighbor_model)
  
  def generate_vector(self, mean_origin, sample, origin_vector):
    features = sample.iloc[2:].values
    vector = Vector(sample.iloc[0], sample.iloc[1], np.subtract(features, mean_origin), None)

    if vector.magnitude > 0:
      vector.angle = self.get_angle(vector, origin_vector)
    
    return vector
  
  def get_angle(self, vector_1, vector_2):
    return math.acos(np.dot(vector_1.vector, vector_2.vector) / (vector_1.magnitude * vector_2.magnitude))
  
  def find_vector_index(self, vector, vector_model, start, end):
    if len(vector_model) == 0 or vector.magnitude == 0:
      return 0
    if start == end:
      if vector_model[start].magnitude == 0 or vector.angle > vector_model[start].angle or \
        (vector.angle == vector_model[start].angle and vector.magnitude >= vector_model[start].magnitude):
        return start + 1
      else:
        return start
    
    mid = int((start + end) / 2)

    if vector_model[mid].magnitude == 0 or vector.angle > vector_model[mid].angle or \
      (vector.angle == vector_model[mid].angle and vector.magnitude >= vector_model[mid].magnitude):
      return self.find_vector_index(vector, vector_model, mid + 1, end)
    else:
      return self.find_vector_index(vector, vector_model, start, mid)
  
  def find_neighbor_index(self, neighbor, nearest_neighbors, start, end):
    if len(nearest_neighbors) == 0:
      return 0
    if start == end:
      return start if neighbor.distance < nearest_neighbors[start].distance else start + 1
    
    mid = int((start + end) / 2)

    if neighbor.distance < nearest_neighbors[mid].distance:
      return self.find_neighbor_index(neighbor, nearest_neighbors, start, mid)
    else:
      return self.find_neighbor_index(neighbor, nearest_neighbors, mid + 1, end)
  
  def find_nearest_neighbors(self, test_vector, model, k):
    # zero magnitude test_vector
    if test_vector.magnitude == 0:
      return model.neighbor_model[:k]

    # initialize variables
    vector_model = model.vector_model
    nearest_neighbors = []
    angle_bound = math.pi
    magnitude_bound = float("inf")
    left_in_bounds = right_in_bounds = True
    right = self.find_vector_index(test_vector, vector_model, 0, len(vector_model) - 1)
    left = right - 1

    # find k-nearest neighbors
    while left_in_bounds or right_in_bounds:
      left_vector = vector_model[left] if left_in_bounds and left >= 0 else None
      right_vector = vector_model[right] if right_in_bounds and right < len(vector_model) else None
      closest_vector = self.get_closest_vector(test_vector, left_vector, right_vector)

      if closest_vector is not None:
        if closest_vector == left_vector:
          left -= 1
        else:
          right += 1

        # check if closest_vector has zero magnitude or is within angle boundaries from origin_vector
        if (closest_vector.magnitude == 0 and test_vector.magnitude <= magnitude_bound) or \
          (test_vector.angle - angle_bound <= closest_vector.angle and closest_vector.angle <= test_vector.angle + angle_bound):
          
          # check if closest_vector is within boundaries from test_vector
          if (closest_vector.magnitude == 0 and test_vector.magnitude <= magnitude_bound) or \
            (self.get_angle(test_vector, closest_vector) <= angle_bound and \
            test_vector.magnitude - magnitude_bound <= closest_vector.magnitude and \
            closest_vector.magnitude <= test_vector.magnitude + magnitude_bound):
            
            # insert neighbor into nearest_neighbors list
            distance = self.get_distance(test_vector, closest_vector)
            neighbor = Neighbor(closest_vector.id, closest_vector.label, distance)
            index = self.find_neighbor_index(neighbor, nearest_neighbors, 0, len(nearest_neighbors) - 1)
            nearest_neighbors.insert(index, neighbor)
            
            # remove farthest neighbor
            if len(nearest_neighbors) > k:
              nearest_neighbors.pop()
            
            # update boundaries
            if len(nearest_neighbors) == k:
              magnitude_bound = nearest_neighbors[-1].distance
              if test_vector.magnitude >= magnitude_bound:
                angle_bound = math.asin(magnitude_bound / test_vector.magnitude)
        else:
          if closest_vector == left_vector:
            left_in_bounds = False
          else:
            right_in_bounds = False
      else:
        left_in_bounds = right_in_bounds = False

    return nearest_neighbors

  def get_closest_vector(self, test_vector, left_vector, right_vector):
    # null vectors
    if left_vector is None and right_vector is None:
      return None
    elif left_vector is None:
      return right_vector
    elif right_vector is None:
      return left_vector

    # zero magnitude vectors
    if left_vector.magnitude == 0:
      return left_vector
    elif right_vector.magnitude == 0:
      return right_vector

    # vectors with both magnitude and direction
    if abs(test_vector.angle - left_vector.angle) < abs(test_vector.angle - right_vector.angle) or \
      (abs(test_vector.angle - left_vector.angle) == abs(test_vector.angle - right_vector.angle) and \
      abs(test_vector.magnitude - left_vector.magnitude) < abs(test_vector.magnitude - right_vector.magnitude)):
      return left_vector
    else:
      return right_vector

  def get_distance(self, test_vector, closest_vector):
    if closest_vector.magnitude == 0:
      return test_vector.magnitude

    angle = self.get_angle(test_vector, closest_vector)

    if angle == 0:
      return abs(test_vector.magnitude - closest_vector.magnitude)
    elif angle == math.pi:
      return test_vector.magnitude + closest_vector.magnitude
    else:
      return math.sqrt(test_vector.magnitude**2 + closest_vector.magnitude**2 - \
        (2 * test_vector.magnitude * closest_vector.magnitude * math.cos(angle)))

  def make_prediction(self, nearest_neighbors):
    # Label dictionary {label: str, count: int}
    labels = {}

    # count occurrences for each label
    for neighbor in nearest_neighbors:
      if neighbor.label in labels:
        labels[neighbor.label] += 1
      else:
        labels[neighbor.label] = 1            

    # find label with max occurrences
    return max(labels.items(), key = itemgetter(1))[0]

if __name__ == "__main__":
  # read parameters
  if len(sys.argv) != 4:
    print("Usage: bknn.py <train_data> <test_data> <k>")
    sys.exit(1)
  
  start_time = time.time()
  train_data = sys.argv[1]
  test_data = sys.argv[2]
  k = sys.argv[3]

  # validate parameters
  errors = False

  try:
    train_data = pd.read_csv(train_data)
  except:
    print("Error: unable to read train data")
    errors = True
  
  try:
    test_data = pd.read_csv(test_data)
  except:
    print("Error: unable to read test data")
    errors = True
  
  try:
    k = int(k)
  except:
    print("Error: k should be an integer")
    errors = True

  # execute algorithm
  if not errors:
    start_cpu_time = time.clock()
    BKNN().execute(train_data, test_data, k)
    print("Ellapsed CPU time: " + str(round(time.clock() - start_cpu_time, 3)) + " seconds")
  
  print("Ellapsed total time: " + str(round(time.time() - start_time, 3)) + " seconds")