import matplotlib.pyplot as plt

y_min = 0
y_max = 100


def plot_raw_results(results, name):
    acc = []
    label = []
    for line in results:
        acc.append(line["accuracy"])
        label.append("{0}, {1}".format(line["fold"], line["number_of_fold"]))
    # d = {"Accuracy %": acc, "Fold %, Numer of fold": label}
    # data_frame = pd.DataFrame(d)
    fig, ax = plt.subplots(figsize=(18, 5))

    ax.bar(label, acc)
    ax.set_ylabel("Accuracy %")
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("Training %, Numer of validation")
    ax.set_title("Accuracy")
    # plt.show()
    fig.savefig(name)
    plt.close()


def plot_results(data_per_class, path):
    average_accuracy = []
    best_accuracy = []
    time_of_classification = []
    time_of_training = []
    classifiers = []
    for clas in data_per_class:
        values = data_per_class[clas]["values"]
        classifiers.append(clas)
        average_accuracy.append(values[0])
        best_accuracy.append(values[1])
        time_of_classification.append(values[2])
        time_of_training.append(values[3])
    print()
    plot_average_accuracy(average_accuracy, classifiers, path + "/Average_accuracy.png")
    plot_best_accuracy(best_accuracy, classifiers, path + "/Best_accuracy.png")
    plot_time_of_classification(time_of_classification, classifiers, path + "/Time_of_classification.png")
    plot_time_of_training(time_of_training, classifiers, path + "/Time_of_training.png")


def plot_average_accuracy(accuracy, labels, path):
    fig, ax = plt.subplots(figsize=(18, 5))

    ax.bar(labels, accuracy)
    ax.set_ylabel("Accuracy %")
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("Classifiers")
    ax.set_title("Average accuracy")
    # plt.show()
    fig.savefig(path)
    plt.close()


def plot_best_accuracy(accuracy, labels, path):
    fig, ax = plt.subplots(figsize=(18, 5))

    ax.bar(labels, accuracy)
    ax.set_ylabel("Accuracy %")
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("Classifiers")
    ax.set_title("Best accuracy")
    # plt.show()
    fig.savefig(path)
    plt.close()


def plot_time_of_classification(accuracy, labels, path):
    fig, ax = plt.subplots(figsize=(18, 5))

    ax.bar(labels, accuracy)
    ax.set_ylabel("Time s")
    ax.set_xlabel("Classifiers")
    ax.set_title("Time of classification")
    # plt.show()
    fig.savefig(path)
    plt.close()


def plot_time_of_training(accuracy, labels, path):
    fig, ax = plt.subplots(figsize=(18, 5))

    ax.bar(labels, accuracy)
    ax.set_ylabel("Time s")
    ax.set_xlabel("Classifiers")
    ax.set_title("Time of training")
    # plt.show()
    fig.savefig(path)
    plt.close()
