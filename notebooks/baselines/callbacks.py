from tensorflow.keras.callbacks import Callback

class NValidationSetsCallback(Callback):
    def __init__(self, validation_sets, verbose=0):
        """
        :param validation_sets:
        dictionary mapping validation set name to validation set generator
        :param verbose:
        verbosity mode, 1 or 0
        """
        super().__init__()
        self.validation_sets = validation_sets
        self.epoch = []
        self.history = {}
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set_name, validation_set in self.validation_sets.items():
            results = self.model.evaluate_generator(validation_set,
                                                    verbose=self.verbose)

            for i, result in enumerate(results):
                valuename = validation_set_name + '_' + self.model.metrics_names[i]
                self.model.history.history.setdefault(valuename, []).append(result)


class GradientLogger(Callback):
    def __init__(self, live_metrics=['loss'], live_gaps=10):
        super().__init__()
        self.metrics = []
        self.batches = 0
        self.live_gaps=live_gaps
        self.live_metrics = live_metrics

    def on_train_begin(self, logs=None):
        for name in self.live_metrics:
            self.add_graph('live_'+name, xlabel='batch')
        for name in self.model.metrics_names:
            self.add_graph(name)

    def on_batch_end(self, batch, logs=None):
        self.batches += 1
        if (self.batches%self.live_gaps) == 0:
            for name in self.live_metrics:
                item = logs[name]
                self.add_data_to_graph(name, str(self.batches), item)
                
    def on_epoch_end(self, epoch, logs=None):
        epoch_str = str(epoch)
        for name, items in self.model.history.history.items():
            if name not in self.metrics:
                self.add_graph(name)
            self.add_data_to_graph(name, epoch_str, items[-1])

    def add_graph(self, name, xlabel='epoch'):
        self.metrics.append(name)
        print('''{"chart": "%s", "axis": "%s"}''' % (name, xlabel))

    def add_data_to_graph(self, name, x, y):
        print('''{"chart": "%s", "y": %s, "x": %s}''' %(name, y, x))
