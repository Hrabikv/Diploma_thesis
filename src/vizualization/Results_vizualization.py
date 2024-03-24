import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_history_of_MLP(history, metric):
    path = "results/MLP_history"
    if not os.path.isdir(path):
        os.mkdir(path)
    # metric = metrics[0]
    # fig, ax = plt.subplots(len(metric))
    # for metric, i in enumerate(metrics):
    #     ax[i].plot(history.history[metric])
    #     ax[i].plot(history.history["val_" + metric])
    #     ax[i].title("model " + metric)
    #     ax[i].ylabel(metric, fontsize="large")
    #     ax[i].xlabel("epoch", fontsize="large")
    #     ax[i].legend(["train", "val"], loc="best")

    plt.plot(history.history[metric])
    # plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train"], loc="best")
    # plt.show()
    # plt.savefig("results/MLP_history/history.png")
    plt.close()


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
    ax.set_xlabel("Training %, Numer of validation")
    ax.set_title("Accuracy")
    # plt.show()
    fig.savefig(name)
    plt.close()
