import numpy as np
import matplotlib.pyplot as plt


def plot_two_results(result_1, result_2=None, experiments=[''], title=''):
    labels = []
    array_1 = []
    array_2 = []

    vals = list(result_1.values())
    labl = list(result_1.keys())
    modules = [labl[i] for i in np.argsort(vals)]

    for module in modules:
        labels.append(module.split('__')[1])
        array_1.append(result_1[module])

    if result_2 is not None:
        for module in modules:
            array_2.append(result_2[module])

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    if result_2 is not None:
        rects1 = ax.bar(x - width / 2, array_1, width, label=experiments[0])
        rects2 = ax.bar(x + width / 2, array_2, width, label=experiments[1])
    else:
        rects1 = ax.bar(x, array_1, width, label=experiments[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation scores: ' + title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    if result_2 is not None:
        autolabel(rects2)

    autolabel(rects1)
    fig.tight_layout()
    plt.ylim([0, 1.2])
    plt.legend(loc='upper left')
    plt.show()


def evaluate_on_dataset(lstm_eval, input_texts, target_texts, exp_names, ref_results, test_set='interpolate', print_res=False):
    results = {}
    for module in input_texts[test_set]:
        metric = lstm_eval.evaluate_model(input_texts[test_set][module], target_texts[test_set][module])
        if print_res:
            print(test_set, module)
            print('metric:', metric, '\n')
        results[module] = metric
    plot_two_results(ref_results, results, experiments=exp_names, title=test_set)

    return results, ref_results
