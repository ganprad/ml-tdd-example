import os

from pytest import mark
from scipy.stats import stats

from mvalidators.data_schema import Constants


@mark.smoke
@mark.linear_regression
class LinearRegressionTests:

    @mark.fit
    def test_fit(self, test_df, test_model):
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        if os.path.exists(test_model.job_parameters.model_file):
            os.remove(test_model.job_parameters.model_file)
        test_model.fit(x, y)
        assert os.path.exists(test_model.job_parameters.model_file)

    @mark.evaluate
    def test_evaluate(self, test_df, test_model):
        """
        Evaluate the model with Area Under the precision and recall curve (AUC).
        Check that the model performance is better that 60%.
        """
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        results = test_model.evaluate(x, y)
        assert results["auc"] > 0.6

    @mark.predict
    def test_predict(self, test_df, test_model):
        """ Statistical hypothesis test to ensure a model
            outputs fall within reasonable ranges after retraining with additional data.
            If the ttest between the model predictions and the ground truth results in a p_val<=5%
            the null hypothesis is rejected. This implies the model has predictive power on this data.
        """
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        preds = test_model.predict(x)
        assert stats.ttest_ind(y, preds)[1] <= 0.05

    @mark.tune
    def test_tune(self, test_df, test_model):
        """Tune model parameters and save tune model.
        Check if the tuned model has been saved at the predefined path."""
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        if os.path.exists(test_model.job_parameters.model_file):
            os.remove(test_model.job_parameters.model_file)
        _ = test_model.tune(x, y)
        assert os.path.exists(test_model.job_parameters.model_file)

    @mark.predict_proba
    def test_predict_proba(self, test_df, test_model):
        """ Statistical hypothesis test to ensure a model
            outputs fall within reasonable ranges after retraining with additional data.
        """
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        preds = test_model.predict_proba(x)
        preds = preds[:, 1]
        t_test = stats.ttest_ind_from_stats(mean1=y.mean(), mean2=preds.mean(), std1=y.std(), std2=preds.std(),
                                            nobs1=y.shape[0], nobs2=preds.shape[0])
        p_val = t_test[1]
        assert p_val <= 0.05
