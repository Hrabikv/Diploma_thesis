
class Representative:

    def __init__(self, value, metric, pivot, label):
        self.value = value
        self.metric = metric
        self.pivot = pivot
        self.label = label

    def get_label(self):
        return self.label

    def get_pivot(self):
        return self.pivot

    def get_metric(self):
        return self.metric

    def get_value(self):
        return self.value
