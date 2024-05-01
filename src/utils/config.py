NUMBER_OF_CLASSES = 3  # 2 - binary classification, 3 - multiclass classification
TRAINING_INFO = 2  # 0 = silent, 1 = progress bar, 2 = one line per epoch
TESTING_INFO = 0  # 0 = silent, 1 = progress bar, 2 = one line per epoch
CLASSIFIERS = ["MLP", "CNN", "LSTM"]  # array of classifiers that will be used.
# Implemented classifiers are: "Statistic", "MLP", "CNN", "LSTM", "Transformer"
TYPE_OF_DATA = "Kodera_29"  # which part of dataset will be pass into classifiers
# "intra-subject" - each subject from both datasets
# "all" - all 41 subjects from both datasets as inter-subject
# "Kodera_29" 29 subjects from Kodera_29 as inter-subject
# "Farabbi_12" 12 subjects from Farabbi_12 as inter-subject
FEATURE_VECTOR = "freq"  # "time" - time-series, "freq" - frequency feature vectors
CUDA_USE = True # true - for use of CUDA GPU, false - if you don't want to use CUDA GPU

