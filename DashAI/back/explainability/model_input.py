"""Helper to move an explainer's data into the model's feature space.

Explainers receive the model input as the task prepared it, exactly as
``predict`` receives it in the prediction job: raw columns, before any model
specific preprocessing. That is what explainers that perturb the input and
query ``model.predict`` need, since ``predict`` applies the model preparation
itself.

Explainers that instead build feature matrices (``to_pandas``) and hand them
to a third party library (SHAP, DiCE, scikit-learn inspection) must work in
the model's own feature space, because those libraries call the model with
plain frames that bypass the model preparation. Such explainers call
:func:`prepare_model_input` on both the background data and the instances so
that both live in the same space.
"""

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset


def as_shap_predictor(model: Any) -> Callable:
    """Wrap ``model.predict`` so SHAP receives a plain function, not a method.

    SHAP suppresses scikit-learn's "X does not have valid feature names"
    warning by blanking ``feature_names_in_`` on whatever object the callable
    is bound to (``shap.utils._legacy.convert_to_model``). It reaches that
    object through ``__self__``, so it only does this when handed a *bound
    method*, and it assumes the attribute is writable.

    That assumption does not hold for estimators that expose
    ``feature_names_in_`` as a read-only ``property`` — a common shape among
    third party wrappers — where the assignment raises ``AttributeError:
    property 'feature_names_in_' ... has no setter`` and the explanation fails
    before it starts.

    Handing over a plain closure instead leaves ``__self__`` absent, so SHAP
    skips that step entirely — a function is SHAP's primary documented
    interface for ``model``. The only thing lost is the suppression of a
    cosmetic scikit-learn warning.

    Parameters
    ----------
    model : Any
        The trained model being explained.

    Returns
    -------
    Callable
        A one-argument function calling ``model.predict`` positionally, the
        same way SHAP calls it today.
    """

    def predict(x):
        return model.predict(x)

    return predict


def prepare_model_input(model: Any, dataset: "DashAIDataset") -> "DashAIDataset":
    """Apply the model's own input preprocessing to a dataset.

    Parameters
    ----------
    model : Any
        The DashAI model being explained.
    dataset : DashAIDataset
        Input features as the task prepared them.

    Returns
    -------
    DashAIDataset
        The dataset in the model's feature space, or the dataset unchanged
        when the model does not define ``prepare_dataset``.
    """
    prepare = getattr(model, "prepare_dataset", None)
    if prepare is None:
        return dataset
    return prepare(dataset, is_fit=False)
