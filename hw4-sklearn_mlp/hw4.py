r'''
 /hw4-sklearn_mlp/hw4.py
 CHANGELOG :
    v2.0.0 - 2021-12-10t19:02
        implemented backward difference system

    v2.0.0 - 2021-12-10t17:04
        combining notebooks

    v1.1.7 - 2021-12-10t17:04
        part012_removing_outliers.ipynb :
            removing ys with Xs

    v1.1.6 - 2021-12-10t13:01
        part015_feature_pruning_selection.ipynb :
            selecting features

    v1.1.5 - 2021-12-10t09:44
        part030_training_the_mlp.ipynb :
            testing out hyperparameters

    v1.1.4 - 2021-12-10t09:35
        part012_removing_outliers.ipynb :
            removing distant sample

    v1.1.3 - 2021-12-10t06:52
        part012_removing_outliers.ipynb :
            removing outliers on features

    v1.1.2 - 2021-12-10t06:52
        part030_training_the_mlp.ipynb :
            measuring accuracy with MLP, but ~5% accurate, very little
                difference between runs

    v1.1.2 - 2021-12-10t00:12
        part020_normalize_features.ipynb :
            completed `main` function
        *.ipynb :
            syntax documentation for functions returning tuples

    v1.1.1 - 2021-12-08t03:20
        part020_normalize_features.ipynb :
            implemented and tested `findFeatureParameters` and
                `normalizeFeatures`
        part010_splitting_the_data.ipynb :
            added `Axis` enum
        Makefile :
            making `part020`, `part010`

    v1.1.0 - 2021-12-08t02:00
        part020_normalize_features.ipynb :
            imported `normalizeFeatures` from `hw3-gradient_descent`

    v1.0.4 - 2021-12-07t23:26
        part010_splitting_the_data.ipynb :
            `main` function, added `removeIds`, `splitLabels` to
                `splitFeaturesLabels`

    v1.0.3 - 2021-12-07t22:50
        part010_splitting_the_data.ipynb :
            documented and cleaned up

    v1.0.2 - 2021-12-07t19:36
        part010_splitting_the_data.ipynb :
            imported `splitFeaturesLabels` from `hw3-gradient_descent`
            fixed keys when checking that the data's IDs are in order
                (originally, it would only check 'trainXy')

    v1.0.1 - 2021-12-06t03:47
        part010_splitting_the_data.ipynb :
            assured that the IDs are succeeding in order

    v1.0.0 - 2021-12-06t03:17
        part010_splitting_the_data.ipynb :
            reading in the data files

    v0.1.1 - 2021-12-05t23:58
        learning_sklearn_mlp.ipynb :
            plotted the results

    v0.1.0 - 2021-12-05t21:12
        .gitignore :
            fixed path for project and `dataset-in/`
        learning_sklearn_mlp.ipynb :
            experimented with the class `sklearn.neural_network.
                MLPClassifier`

    v0.0.1 - 2021-12-05t14:16
        CHANGELOG :
            formatted CHANGELOG so far

    v0.0.0 - 2021-12-05t12:48
        .gitingore :
            moved dataset into its own folder, which gets .gitignore'd
        CHANGELOG :
            added CHANGELOGs

    v(-1).2.0 - 2021-12-05t03:57
        .gitattribute :
            combined both filters for *.ipynb and renamed the
                combined filter in to `ipynb`
            now filtering successfully

    v(-1).1.0 - 2021-12-05t02:14
        .gitattribute :
            added filter for cell ID
                (note: two filters on one file do not work)
        HelloWorld.ipynb
            added the cells[1]

    v(-1).0.0 - 2021-12-05t01:55
        .gitingore :
            releases, compressed files, datasets (*.csv *.docx),
                binaries, Jupyter notebook checkpoints,
                exported scripts
        .gitattribute :
            applying the execution count filter
        HelloWorld.ipynb :
            hello world project in Jupyter notebook
 '''

import numpy as np  # for np.ndarray
from enum import Enum # to superclass Axis

def main():
    data = putDataFromFiles(DATA_DIR, DATA_FILENAMES, {})
    print(r'data', data[r'trainXy'])
    edgeArray = removeConflictingFeatures(data[r'trainXy'])
    print(r'edgeArray', edgeArray)
    print(r'edgeArray', removeConflictingFeatures(np.array([[1,2,3],[4,5,6],[1,2,3]])))

# constants
DATA_DIR = r'dataset-in/'   # directory holding the data files
# names of files holding dataset
DATA_FILENAMES = {\
                  r'trainXy': r'train.csv',\
                  r'testX':   r'test.csv',\
                  r'test_y':  r'sample.csv',\
                 }
DELIMITER = r','    # used to separate values in DATA_FILENAME

# array for backwards difference vector
DIFF_V = np.array([1, -1])

# axes of numpy arrays
class Axis(Enum):
    COLS = 0
    ROWS = 1
# class Axis(Enum)

def putDataFromFiles(prefix, srcFilenames, dest, delimiter=DELIMITER):
    r'''
     Puts the data from the files represented by the source filenames
     into the given destination dictionary.

     Although this function leaves the keys as abstract, it is expected
     that keys represent the type of data (trainXy, testX, test_y)
     contained in each file whose name the key maps to.

     @param srcFilenames : dict<TKey,str> = dictionary mapping to
         filenames containing the data
     @param dest : dict<? super TKey,np.ndarray> = dictionary to
         which to map the arrays
     @return `destDict`
     '''
    # loop through each mapping to the name of the file
    for key, file in srcFilenames.items():
        # generate the arrays from the data contained therein
        dest[key] = np.genfromtxt(
            fr'{prefix}{file}', delimiter=delimiter,
            skip_header=True, dtype=np.float64)
    # for key, file in srcFilenames.items()
    return dest
# def putDataFromFiles(srcFilenames, dest)

def removeConflictingFeatures(multidataset):
    r'''
     Removes conflicting features.
     @param multidataset : np.ndarray = dataset that may contain
         duplicates.
     @return the dataset containing no duplicates
     '''
    sortedData = sortRows(multidataset)
    edgeArray = backwardDifference(sortedData, sortedData.shape[0])
    return edgeArray
# def removeConflictingFeatures(dataset)

def sortRows(unsorted):
    r'''
     Sorts an array by each of its rows.
     @param unsorted : np.ndarray = array to sort
     '''
    # convert array into sorted list of tuples
    sortedTuples = sorted(map(tuple, unsorted))
    # convert back to an array
    sortedArray = np.array(sortedTuples)
    return sortedArray
# def sortRows(array)

def backwardDifference(array, nRows):
    r'''
     Applies a backward difference system to each column of the
     specified array.
     @param array : np.ndarray = to which to apply the system
     @param nRows : int = the nuber of rows in the array
     @return the array with the backward difference applied
     '''
    # covolve each row and comprehend into a list
    conv = [np.convolve(array[:,k], DIFF_V) for k in range(array.shape[1])]
    # convert to a numpy array and transpose
    convArray = np.array(conv).T
    # given that array has R rows,
    # the convolved array has (R + 2 - 1) = (R + 1) columns.
    # so split off the last row
    (edgeArray, _) = np.split(convArray,
                         # all rows except for last
                         (nRows,), axis=Axis.COLS.value)
    return edgeArray
# def backwardDifference(array)

def splitFeaturesLabels(dataset, removeIds = False, splitLabels = True):
    r'''
     Divides the dataset into features and labels.
     @syntax (features, labels)
         = splitFeaturesLabels(dataset, removeIds, splitLabels)
     @param dataset : np.ndarray = the dataset to divide
     @param removeIds : bool = whether to remove an initial ID column
     @return a tuple containing the feature arrays and label vector
     '''
    # get the number of rows and columns
    (num_rows, num_cols) = dataset.shape
    # split each row of the dataset
    (_, features, M_label_scalars) = \
        np.split(dataset,
                 # skip column 1 if removing IDs
                 ((1 if removeIds else 0),
                  # if splitting labels, stop 1 column early
                  (num_cols - (1 if splitLabels else 0))
                 ), axis=Axis.ROWS.value)
    # convert to a vector
    if (splitLabels):
        labels = M_label_scalars.reshape((num_rows,))
    else:
        labels = M_label_scalars
    # if (splitLabels)
    return (features, labels)
# def splitFeaturesLabels(dataset)

# if main module
if __name__ == "__main__":
    main()
# if __name__ == "__main__"

