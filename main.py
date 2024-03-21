from preprocessing import ETL
from classification.statistical_approach.Statistic_classification import StatisticClassification
from classification.Cross_Validation import cross_validation
from classification.mlp.MLP import MLP

if __name__ == '__main__':
    data, labels = ETL.load_data()
    # plot_data_of_all_subject(data, labels, "")
    # plot_data_sample(data_sample=np.concatenate(data[0][0]), label="left")
    # plot_data_sample(data_sample=np.concatenate(data[0][2]), label="rest")
    # plot_data_sample(data_sample=np.concatenate(data[0][66]), label="right")
    # c = StatisticalClassification(data=data[0], labels=labels[0])
    # c.compute_class_representative()
    # c.compute_pivots_value_of_representations()
    # classifiers = [MLP()]
    classifiers = [StatisticClassification(), MLP()]
    cross_validation(vectors=data[0], labels=labels[0], classifiers=classifiers, subject="1")
    print()
