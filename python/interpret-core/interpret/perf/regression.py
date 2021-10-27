# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import unify_data, gen_name_from_class, unify_predict_fn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from ..glassbox.ebm.bin import unify_data2

class RegressionPerf(ExplainerMixin):
    """ Produces variety of regression metrics (including RMSE, R^2, etc)."""

    available_explanations = ["perf"]
    explainer_type = "perf"

    def __init__(self, predict_fn, feature_names=None, feature_types=None, **kwargs):
        """ Initializes class.

        Args:
            predict_fn: Function of blackbox that takes input, and returns prediction.
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Currently unused. Due for deprecation.
        """
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.kwargs = kwargs

    def explain_perf(self, X, y, name=None):
        """ Produces regression metrics.

        Args:
            X: Numpy array for X to compare predict function against.
            y: Numpy vector for y to compare predict function against.
            name: User-defined explanation name.

        Returns:
            An explanation object.
        """
        if name is None:
            name = gen_name_from_class(self)

        X0 = X
        y0 = y
        w0 = None
        feature_types0 = self.feature_types
        feature_names0 = self.feature_names


        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types, missing_data_allowed=True
        )


        X1 = X
        y1 = y
        w1 = None
        feature_types1 = self.feature_types
        feature_names1 = self.feature_names
        are_classifier = None if y0 is None else not issubclass(np.array(y0).dtype.type, np.floating)

        if feature_types0 is not None:
            feature_types0 = ["nominal" if feature_type == "categorical" else feature_type for feature_type in feature_types0]
        feature_types1 = ["nominal" if feature_type == "categorical" else feature_type for feature_type in feature_types1]
        X2, y2, w2, feature_names2, feature_types2 = unify_data2(are_classifier, X0, y0, w0, feature_names0, feature_types0)

        if y1 is not None:
            if not np.array_equal(y1, y2):
                raise NotImplementedError("oh no EBM y!")

        if w0 is not None:
            if not np.array_equal(w1, w2):
                raise NotImplementedError("oh no EBM w!")

        if feature_names1 != feature_names2:
            raise NotImplementedError("oh no EBM feature_names!")

        if feature_types1 != feature_types2:
            raise NotImplementedError("oh no EBM feature_types!")

        X1 = X1.astype(np.object_)
        for idx in range(len(feature_types1)):
            if feature_types1[idx] == 'continuous':
                X1[:, idx] = X1[:, idx].astype(np.float64).astype(np.object_)
        X1 = X1.astype(np.unicode_)
        X2 = X2.astype(np.unicode_)
        if not np.array_equal(X1, X2):
            raise NotImplementedError("oh no EBM X!")

        predict_fn = unify_predict_fn(self.predict_fn, X)
        scores = predict_fn(X)

        mse = mean_squared_error(y, scores)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, scores)
        r2 = r2_score(y, scores)
        residuals = y - scores

        # abs_residuals = np.abs(y - scores)
        counts, values = np.histogram(residuals, bins="doane")

        overall_dict = {
            "type": "perf_curve",
            "density": {"names": values, "scores": counts},
            "scores": scores,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "residuals": residuals,
        }
        internal_obj = {"overall": overall_dict, "specific": None}

        return RegressionExplanation(
            "perf",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
        )


class RegressionExplanation(ExplanationMixin):
    """ Produces explanation specific to regression metrics."""

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        """ Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """
        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        """ Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.

        Returns:
            A serializable dictionary.
        """
        if key is None:
            return self._internal_obj["overall"]
        return None

    def visualize(self, key=None):
        """ Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ..visual.plot import plot_density

        data_dict = self.data(key)
        if data_dict is None:
            return None

        rmse = data_dict["rmse"]
        r2 = data_dict["r2"]

        title = "{0} <br> RMSE = {1:.2f}" + " | R<sup>2</sup> = {2:.2f}"
        title = title.format(self.name, rmse, r2)
        density_fig = plot_density(
            data_dict["density"], title=title, xtitle="Residuals", ytitle="Density"
        )
        return density_fig
