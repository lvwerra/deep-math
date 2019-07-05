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