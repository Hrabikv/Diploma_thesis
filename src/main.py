from preprocessing import ETL
from classification.Cross_Validation import cross_validation
from classification.statistical_approach.Statistic_classification import StatisticClassification
from classification.cnn.CNN import CNN
from classification.mlp.MLP import MLP
from classification.lstm.LSTM import LSTM

# binary LSTM subject 21 done
# binary CNN, MLP subject 35 done


def tests_all_subjects(data, labels, classifiers):
    i = 1
    for subject, label in zip(data, labels):
        cross_validation(vectors=subject, labels=label, classifiers=classifiers, subject="{}".format(i))
        i = i + 1


if __name__ == '__main__':
    d, l = ETL.load_data()
    classifiers_array = [StatisticClassification(), MLP(), CNN()]
    # cross_validation(vectors=d[8], labels=l[8], classifiers=classifiers_array, subject="9")
    tests_all_subjects(d, l, classifiers_array)
    print()
