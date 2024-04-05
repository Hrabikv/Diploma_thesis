NUMBER_OF_CLASSES = 3  # 2 - binary classification, 3 - multiclass classification
TRAINING_INFO = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch
TESTING_INFO = 0  # 0 = silent, 1 = progress bar, 2 = one line per epoch
CLASSIFIERS = ["Transformer"]  # array of classifiers that will be used.
# Implemented classifiers are: "Statistic", "MLP", "CNN", "LSTM", "Transformer"
TYPE_OF_DATA = "intra-subject"  # which part of dataset will be pass into classifiers
# "intra-subject" - each subject from both datasets
# "all" - all subject from both datasets as inter-subject
# "Kodera_29" 29 subjects from Kodera_29 as inter-subject
# "Farabbi_12" 12 subjects from Farabbi_12 as inter-subject

