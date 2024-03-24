from preprocessing import ETL
from classification.Cross_Validation import cross_validation
from classification.cnn.CNN import CNN
from classification.mlp.MLP import MLP


def tests_all_subjects(data, labels, classifiers):
    i = 1
    for subject, label in zip(data, labels):
        cross_validation(vectors=subject, labels=label, classifiers=classifiers, subject="{}".format(i))
        i = i + 1


if __name__ == '__main__':
    d, l = ETL.load_data()
    classifiers_array = [CNN(), MLP()]
    # cross_validation(vectors=d[8], labels=l[8], classifiers=classifiers_array, subject="9")
    tests_all_subjects(d, l, classifiers_array)
    print()
