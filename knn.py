import sys, time
import pandas as pd
from operator import itemgetter

class Neighbor:
  def __init__(self, id, label, distance):
    self.id = id
    self.label = label
    self.distance = distance

class KNN:
  def execute(self, train_data, test_data, k):
    # track correct predictions
    correct = 0

    # classify test data
    for i in range(len(test_data)):
      nearest_neighbors = self.find_nearest_neighbors(test_data.iloc[i], train_data, k)
      prediction = self.make_prediction(nearest_neighbors)

      if prediction == test_data.iloc[i, 1]:
        correct += 1

    # calculate and display accuracy
    print("Accuracy: " + str(correct) + " out of " + str(len(test_data)) + " (" + str(round((float(correct) / len(test_data)) * 100, 2)) + "%)")

  def find_nearest_neighbors(self, test_sample, train_data, k):
    # list of neighbors
    neighbors = []

    # calculate distance to all train samples
    test_features = test_sample.iloc[2:].values

    for i in range(len(train_data)):
      train_features = train_data.iloc[i, 2:].values
      distance = sum((test_features - train_features)**2)**0.5
      neighbors.append(Neighbor(train_data.iloc[i, 0], train_data.iloc[i, 1], distance))
    
    # find k-nearest neighbors
    neighbors.sort(key = lambda neighbor: neighbor.distance)
    return neighbors[:k]
  
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
    print("Usage: knn.py <train_data> <test_data> <k>")
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
    KNN().execute(train_data, test_data, k)
    print("Ellapsed CPU time: " + str(round(time.clock() - start_cpu_time, 3)) + " seconds")
  
  print("Ellapsed total time: " + str(round(time.time() - start_time, 3)) + " seconds")