import matplotlib.pyplot as plt


def plot_raw_results(results):
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
    ax.set_xlabel("Fold %, Numer of fold")
    ax.set_title("Accuracy")
    # plt.show()
    return fig
