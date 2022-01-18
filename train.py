import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns
from sklearn import metrics

from mvalidators.data_schema import Constants
from mvalidators.linear_regression_model_schema import HyperParam, ModelParam, JobParam, OptunaCVParam
from mwrapper.linear_regression_model import LinearRegressionModel

sns.set_theme(context="notebook", style="ticks", palette="colorblind")

df = pandas.read_csv(Constants().data_file)
hyper_parameters = HyperParam()
model_parameters = ModelParam()
job_parameters = JobParam()
optuna_cv_parameters = OptunaCVParam()
model = LinearRegressionModel(hyper_parameters, model_parameters, job_parameters, optuna_cv_parameters)

if __name__ == "__main__":
    output_dict, results_dict = model.tune(df.drop("is_bad", axis=1), df["is_bad"].values)

    fpr = results_dict["fpr"]
    tpr = results_dict["tpr"]
    thresholds = results_dict["thresholds"]
    y = results_dict["target"]
    probas = results_dict["probas"][:, 1]

    auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve: Baseline Logistic Regression")
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.text(x=0.8, y=0.1, s=f"ROC curve area = {auc:.2f}", ha="center", va="center")
    plt.legend(["model", "random guess"])
    plt.savefig("../../results/sklearn_logistic_regression_roc.png")
    plt.show()

    sweep = ((thresholds <= 1.0).astype(float) * (thresholds > 0.0).astype(float)).astype(float)
    sweep_thresholds = thresholds[sweep * thresholds > 0]
    recall_scores = [metrics.recall_score(y_pred=(probas > th).astype(float), y_true=y) for th in sweep_thresholds]
    precision_scores = [metrics.precision_score(y_pred=(probas > th).astype(float), y_true=y) for th in
        sweep_thresholds]

    plt.figure()
    d = {th: s for th, s in zip(sweep_thresholds, precision_scores)}
    precision_ranked_thresholds = [d[key] for key in sorted(d.keys())]
    plt.plot(sorted(list(d.keys())), precision_ranked_thresholds)
    d = {th: s for th, s in zip(sweep_thresholds, recall_scores)}
    recall_ranked_thresholds = [d[key] for key in sorted(d.keys())]
    plt.plot(sorted(list(d.keys())), recall_ranked_thresholds)

    plt.axvline(x=y.mean(), color="darkred", lw=0.5, linestyle="--")
    plt.text(x=y.mean(), y=0.5, s=f"target mean: {y.mean():.2f}", rotation=90, fontsize=10, ha="center", va="center")

    min_balance = numpy.inf
    for idx, th in enumerate(sweep_thresholds):
        balance = numpy.abs(recall_scores[idx] - precision_scores[idx])
        if (balance < min_balance) and (balance > 0.0):
            min_balance = balance
            min_balance_idx = idx
    balance_threshold = sweep_thresholds[min_balance_idx]

    plt.axvline(x=balance_threshold, lw=0.5, color="darkred", linestyle="--")
    plt.text(x=balance_threshold, y=0.5, s=f"balance: {balance_threshold:.2f}", rotation=90, fontsize=10, ha="center",
        va="center", )

    plt.xlabel("thresholds")
    plt.ylabel("scores")
    plt.legend(["precision", "recall"])
    plt.grid(True, axis="both")
    plt.title("Trade-off curves: Baseline Logistic Regression ")
    plt.savefig("../../results/sklearn_logistic_regression_tradeoff_sweep.png")
    plt.show()

    # TODO  # Adding post-processing plotting utils.
