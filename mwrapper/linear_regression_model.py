from typing import Dict

import optuna
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn import metrics
from mwrapper.basewrappermodel import BaseWrapperModel
from mwrapper.utils import drop_columns, preprocess_emp_length, preprocess_categorical_columns, preprocess_purpose_cat, \
    fillnans, oh_encode, minmax, make_categorical_columns
from mvalidators.data_schema import InputDataFrameSchema, PreprocessedDataSchema
from mvalidators.constants import Constants
from mvalidators.linear_regression_model_schema import HyperParam, ModelParam, JobParam, OptunaCVParam, \
    TrainedModelExists


def get_hp_distributions(hyper_params: HyperParam) -> Dict:
    """
    This function takes in hyper parameters specified by the HyperParams type and assigns an appropriate sampling
    distribution for the hyper-parameter.

    :param hyper_params: HyperParams
    :return: Dict
    """
    hp_distributions = {}
    param_dict = hyper_params.__dict__
    for name in param_dict.keys():
        param = param_dict[name]
        if type(param.value) == float:
            hp_distributions[param.name] = optuna.distributions.LogUniformDistribution(low=param.min, high=param.max)
        elif type(param.value) == int:
            hp_distributions[param.name] = optuna.distributions.IntLogUniformDistribution(low=param.min, high=param.max)
    return hp_distributions


class LinearRegressionModel(BaseWrapperModel):
    """

    A wrapper for scikit-learn Logistic regression model with preprocessing functionality for the
    Lending Club dataset.

    Parameters
    ----------

    hyper_parameters: HyperParam
        A pydantic datatype that specifies the set of the model hyper parameters to tune for the Lending Club dataset.

    model_parameters: ModelParam
        A pydantic datatype that specifies settings of model parameters.

    job_parameters: JobParam
        A pydantic datatype that specifies job related parameters.

    optuna_cv_parameters: OptunaCVParam
        A pydantic datatype that specifies parameter settings for Optuna based cross-validation.

    """

    def __init__(
        self,
        hyper_parameters: HyperParam,
        model_parameters: ModelParam,
        job_parameters: JobParam,
        optuna_cv_parameters: OptunaCVParam,
    ):
        self.hyper_parameters = hyper_parameters
        self.solver_parameters = model_parameters
        self.job_parameters = job_parameters
        self.optuna_cv_parameters = optuna_cv_parameters
        self.logistic_model_params = dict(
            max_iter=hyper_parameters.max_iter.value,
            C=hyper_parameters.C.value,
            l1_ratio=hyper_parameters.l1_ratio.value,
            solver=model_parameters.solver,
            fit_intercept=model_parameters.fit_intercept,
            warm_start=model_parameters.warm_start,
            multi_class=model_parameters.multi_class,
            penalty=model_parameters.penalty,
            class_weight=model_parameters.class_weight,
            random_state=job_parameters.random_state,
            n_jobs=job_parameters.n_jobs,
            verbose=job_parameters.verbose,
        )
        self.hp_distributions = get_hp_distributions(hyper_parameters)
        self.constants = Constants()

        if self.optuna_cv_parameters.n_repeats >= 1:
            self.kf = RepeatedStratifiedKFold(
                n_splits=optuna_cv_parameters.n_splits,
                n_repeats=optuna_cv_parameters.n_repeats,
                random_state=job_parameters.random_state,
            )
        else:
            self.kf = StratifiedKFold(
                n_splits=optuna_cv_parameters.n_splits, random_state=job_parameters.random_state, shuffle=True
            )
        self.scorer = metrics.get_scorer(model_parameters.metric)

    def preprocess(self, x):
        """
        Preprocessing function.

        :param x:pd.DataFrame
        :return:pd.DataFrame
        """

        valid_input_df = InputDataFrameSchema.validate(x)
        preprocessed_df = drop_columns(valid_input_df, self.constants.drop)
        preprocessed_df = preprocess_emp_length(preprocessed_df)
        preprocessed_df = preprocess_categorical_columns(preprocessed_df, self.constants.one_hot)
        preprocessed_df = preprocess_purpose_cat(preprocessed_df)

        for col in preprocessed_df.columns:
            preprocessed_df = fillnans(preprocessed_df, col)
        assert not pd.isnull(preprocessed_df).values.sum() > 0
        return preprocessed_df

    def train_encode(self, x):
        """
        Train encoders for mvalidators.
        This function trains a minmax encoder and an one hot encoder.
        It validates the encoded output with the preprocessed schema model.

         :param x: pd.DataFrame
        :return:  pd.DataFrame
        """
        x, enc = minmax(x, self.constants.minmax)
        x, oh = oh_encode(x, self.constants.one_hot)
        df = PreprocessedDataSchema.validate(x)
        assert not pd.isnull(df).values.sum() > 0
        joblib.dump(enc, self.constants.minmax_encoder_filename)
        joblib.dump(oh, self.constants.onehot_encoder_filename)
        return df

    def predict_encode(self, x):
        """
        Encode mvalidators for predictions.
        This function fetches trained and saved encoders and uses them to encode
        mvalidators for model predictions.
        The output is validated using the preprocessed schema model.

        :param x: pd.DataFrame
        :return: pd.DataFrame
        """
        oh = joblib.load(self.constants.onehot_encoder_filename)

        x[self.constants.minmax] = minmax.transform(x[self.constants.minmax])
        x_one_hot = oh.transform(x[self.constants.one_hot])
        x_one_hot = pd.DataFrame(x_one_hot.todense())
        x_one_hot = make_categorical_columns(x_one_hot, categories=oh.categories_, columns=self.constants.one_hot)
        x = x.drop(columns=self.constants.one_hot)
        x = pd.concat([x, x_one_hot], axis=1)
        return PreprocessedDataSchema.validate(x)

    def fit(self, x, y):
        """
        Fit and save the logistic regression model given training dataset x and targets y.

        :param x: pd.Dataframe
        :param y: numpy.ndarray
        :return: None
        """
        model = LogisticRegression(**self.logistic_model_params)
        x = self.preprocess(x)
        x = self.train_encode(x)
        model.fit(x, y)
        joblib.dump(model, self.job_parameters.model_file)

    def predict(self, x):
        """
        Predict target "is_bad" outcomes using the scikit-learn logistic regression model.
        This function predicts target variables using trained logistic regression model.

        :param x: pd.DataFrame
        :return: numpy.ndarray
        """

        x = self.preprocess(x)
        x = self.predict_encode(x)
        TrainedModelExists.validate({"model_file": self.job_parameters.model_file})
        model = joblib.load(self.job_parameters.model_file)
        return model.predict(x)

    def predict_proba(self, x):
        """
        Predict probabilities of target "is_bad" outcomes using the scikit-learn logistic regression model.
        This function predicts target probabilities using a trained logistic regression model.

        :param x: pd.DataFrame
        :return: numpy.ndarray
        """
        x = self.preprocess(x)
        x = self.predict_encode(x)
        TrainedModelExists.validate({"model_file": self.job_parameters.model_file})
        model = joblib.load(self.job_parameters.model_file)
        return model.predict_proba(x)

    def evaluate(self, x, y):
        """
        Evaluate f1-score and log-loss for given mvalidators x and targets y.

        :param x: pd.DataFrame
        :param y: numpy.ndarray
        :return: Dict
        """
        preds = self.predict(x)
        logprobs = self.predict_proba(x)
        f1_score = metrics.f1_score(y, preds)
        logloss = metrics.log_loss(y, logprobs[:, 1])
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y, y_score=logprobs[:, 1])
        auc = metrics.auc(fpr, tpr)
        return {"f1_score": f1_score, "logloss": logloss, "auc": auc}

    def tune(self, x, y):
        """
        Tune logistic regression model hyper-parameters using cross-validation for best f1-score.
        Since the model parameters have been optimized for performance on a specific dataset, this tuned model is
        expected to be used on mvalidators originating from the SAME mvalidators generating process or environment.

        :param x: pd.DataFrame
        :param y: numpy.ndarray
        :return:Dict
        """
        model = LogisticRegression(**self.logistic_model_params)
        hp_search = optuna.integration.OptunaSearchCV(
            estimator=model,
            cv=self.kf,
            scoring=self.scorer,
            param_distributions=self.hp_distributions,
            refit=self.optuna_cv_parameters.refit,
            return_train_score=self.optuna_cv_parameters.return_train_score,
            n_trials=self.optuna_cv_parameters.n_trials,
            n_jobs=self.job_parameters.n_jobs,
            random_state=self.job_parameters.random_state,
            verbose=self.job_parameters.verbose,
        )

        x = self.preprocess(x)
        x = self.train_encode(x)
        hp_search.fit(x, y)
        y_pred = hp_search.predict(x)
        probas = hp_search.predict_proba(x)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y, y_score=probas[:, 1], pos_label=1)
        best_params = hp_search.best_estimator_.get_params()
        f1_score = metrics.f1_score(y, y_pred)
        logloss = metrics.log_loss(y, probas[:, 1])
        joblib.dump(hp_search.best_estimator_, self.job_parameters.model_file)
        return (
            {**best_params, "scores": {"f1_score": f1_score, "logloss": logloss}},
            {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "target": y, "probas": probas},
        )

if __name__ == '__main__':
    df = pd.read_csv("../data/data.csv")
    hyper_parameters = HyperParam()
    model_parameters = ModelParam()
    job_parameters = JobParam()
    optuna_cv_parameters = OptunaCVParam()
    m = LinearRegressionModel(hyper_parameters, model_parameters, job_parameters, optuna_cv_parameters)
    preprocessed_df = m.preprocess(df)





