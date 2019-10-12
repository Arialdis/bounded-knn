# Bounded KNN

### Overview
The Bounded KNN is an optimization of the [K-Nearest Neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). It uses vector space models to improve the nearest neighbors search process by dynamically restricting the search area. Therefore, reducing the number of comparisons necessary to find the nearest neighbors for the majority of samples. The experimental results demonstrate significant performance improvements without degrading the algorithm’s accuracy.

For more details, see the publication at [SpringerLink](https://link.springer.com/chapter/10.1007%2F978-3-030-12388-8_43).

**Note:** The published research paper uses the experimental results of an older Java implementation of the algorithms. So, those numbers will differ from the Python results at the bottom of this document.

### Setup
This repo includes sample datasets for initial testing.
```sh
$ git clone https://github.com/Arialdis/bounded-knn.git
$ cd bounded-knn
```

To run the traditional KNN algorithm on the sample datasets using a K value of 5:
```sh
$ python knn.py sample-data/train-950.csv sample-data/test-50.csv 5
```

To run the Bounded KNN algorithm on the sample datasets using a K value of 5:
```sh
$ python bknn.py sample-data/train-950.csv sample-data/test-50.csv 5
```

### Results
Comparison of the traditional KNN vs. Bounded KNN using the sample datasets (950 training samples and 50 test samples).

|  K  | Traditional Accuracy | Bounded Accuracy | Traditional Run-Time | Bounded Run-Time |
|:---:|:--------------------:|:----------------:|:--------------------:|:----------------:|
|  1  |        88.00%        |      88.00%      |         18.33        |       0.865      |
|  3  |        94.00%        |      94.00%      |        18.393        |       0.959      |
|  5  |        90.00%        |      90.00%      |        18.214        |       1.045      |
|  7  |        88.00%        |      88.00%      |        18.217        |       1.04       |
|  9  |        88.00%        |      88.00%      |        18.407        |       1.082      |
|  11 |        86.00%        |      86.00%      |        18.315        |       1.155      |
|  13 |        84.00%        |      84.00%      |        18.414        |       1.098      |
|  15 |        82.00%        |      82.00%      |         18.71        |       1.122      |
| AVG |        87.50%        |      87.50%      |        18.375        |       1.046      |

*Run-time measured in seconds.*