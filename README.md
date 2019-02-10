# automatic-octo-broccoli

## Installation

Do `pip install numpy` if you don't have numpy

## Running the code

### Ten-Fold Cross validation

For the ten fold cross validation, run
```
python3 cross_validation.py <PATH_TO_DATA_FILE> <WILL_BE_PRUNED(true/false)>
```
e.g.
```
python3 cross_validation.py co395-cbc-dt/wifi_db/clean_dataset.txt true
```
The above will build a pruned decision tree with the clean dataset and prints the result of evaluate with the test set.
