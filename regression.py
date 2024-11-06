from xai_components.base import InArg, OutArg, Component, xai_component
from IPython.utils import capture

@xai_component(color="blue")
class SetupRegression(Component):
    """
    Initializes the training environment and creates the transformation pipeline.
    Setup must be called before executing any other component.

    ##### inPorts:
    - in_dataset (any): Shape (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.
    - target (str): Name of the target column to be passed in as a string. The target variable can be either binary or multiclass.
    - train_size_fraction (float): Proportion of the dataset to be used for training and validation. Should be between 0.0 and 1.0.
    - transform_target (bool): When set to True, target variable is transformed using the method defined in transform_target_method param. Target transformation is applied separately from feature transformations.
    - transform_target_method (str): 'quantile' and 'yeo-johnson' methods are supported.
    - normalize (bool): When set to True, it transforms the numeric features by scaling them to a given range.
    - transformation (bool): When set to True, it applies the power transform to make data more Gaussian-like.
    - remove_multicollinearity (bool): When set to True, features with the inter-correlations higher than the defined threshold are removed.
    - multicollinearity_threshold (float): Threshold for correlated features. Ignored when remove_multicollinearity is not True.
    - bin_numeric_features (any): To convert numeric features into categorical. It takes a list of strings with column names that are related.
    - group_features (any): When the dataset contains features with related characteristics, group_features parameter can be used for feature extraction. It takes a list of strings with column names that are related.
    - ignore_features (list): Ignore_features param can be used to ignore features during model training. It takes a list of strings with column names that are to be ignored.
    - seed (int): You can use random_state for reproducibility.
    - log_experiment (bool): Logging setup and training.
    - experiment_name (str): Name of the experiment for logging.
    - use_gpu (bool): Whether to use GPU for training.
    """
    in_dataset: InArg[any]
    target: InArg[str]
    train_size_fraction: InArg[float]
    transform_target: InArg[bool]
    transform_target_method: InArg[str]
    normalize: InArg[bool]
    transformation: InArg[bool]
    remove_multicollinearity: InArg[bool]
    multicollinearity_threshold: InArg[float]
    bin_numeric_features: InArg[any]
    group_features: InArg[any]
    ignore_features: InArg[list]
    seed: InArg[int]
    log_experiment: InArg[bool]
    experiment_name: InArg[str]
    use_gpu: InArg[bool]

    def __init__(self):
        super().__init__()
        self.target.value = 'default'
        self.train_size_fraction.value = 1.0
        self.transform_target.value = False
        self.transform_target_method.value = 'quantile'
        self.normalize.value = False
        self.transformation.value = False
        self.remove_multicollinearity.value = False
        self.multicollinearity_threshold.value = 0.9
        self.experiment_name.value = 'default'
        self.use_gpu.value = False

    def execute(self, ctx) -> None:
        from pycaret.regression import setup, models

        with capture.capture_output() as captured:
            setup(
                data=self.in_dataset.value,
                target=self.target.value,
                train_size=self.train_size_fraction.value,
                transform_target=self.transform_target.value,
                transform_target_method=self.transform_target_method.value,
                normalize=self.normalize.value,
                transformation=self.transformation.value,
                remove_multicollinearity=self.remove_multicollinearity.value,
                multicollinearity_threshold=self.multicollinearity_threshold.value,
                bin_numeric_features=self.bin_numeric_features.value,
                group_features=self.group_features.value,
                ignore_features=self.ignore_features.value,
                session_id=self.seed.value,
                log_experiment=self.log_experiment.value,
                experiment_name=self.experiment_name.value,
                use_gpu=self.use_gpu.value
            )
        captured.show()

        print("List of the Available Regression models: ")
        print(models())

@xai_component(color="firebrick")
class CompareModelsRegression(Component):
    """
    Trains and evaluates performance of all estimators available in the model library using cross validation.
    The output of this component is a score grid with average cross-validated scores.

    ##### inPorts:
    - sort_by (str): The sort order of the score grid.
    - exclude (list): To omit certain models from training and evaluation, pass a list containing model id in the exclude parameter.
    - num_top (int): Number of top_n models to return.
    
    ##### outPorts:
    - top_models (any): List of top models.
    """
    sort_by: InArg[str]
    exclude: InArg[list]
    num_top: InArg[int]
    top_models: OutArg[any]

    def __init__(self):
        super().__init__()
        self.sort_by.value = 'R2'
        self.num_top.value = 1

    def execute(self, ctx) -> None:
        from pycaret.regression import compare_models

        with capture.capture_output() as captured:
            best_model = compare_models(sort=self.sort_by.value, exclude=self.exclude.value, n_select=self.num_top.value)
        captured.show()
        print('Best ' + str(self.num_top.value) + ' Model:', best_model)

        self.top_models.value = best_model

@xai_component(color="orange")
class CreateModelRegression(Component):
    """
    Trains and evaluates the performance of a given estimator using cross validation.
    The output of this component is a score grid with CV scores by fold.

    ##### inPorts:
    - model_id (str): ID of an estimator available in model library or pass an untrained model object consistent with scikit-learn API.
    - num_fold (int): Controls cross-validation. If None, the CV generator in the fold_strategy parameter of the setup function is used.
    
    ##### outPorts:
    - out_created_model (any): Trained Model object.
    """
    model_id: InArg[str]
    num_fold: InArg[int]
    out_created_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.model_id.value = 'lr'
        self.num_fold.value = 10

    def execute(self, ctx) -> None:
        from pycaret.regression import create_model

        with capture.capture_output() as captured:
            created_model = create_model(estimator=self.model_id.value, fold=self.num_fold.value)
        captured.show()
        print(created_model)

        self.out_created_model.value = created_model

@xai_component(color="salmon")
class TuneModelRegression(Component):
    """
    Tunes the hyperparameters of a given model. The output of this component is
    a score grid with CV scores by fold of the best selected model based on optimize parameter.

    ##### inPorts:
    - in_model (any): Trained model object.
    - optimize (str): Metric name to be evaluated for hyperparameter tuning.
    - early_stopping_patience (int): Maximum number of epochs to run for each sampled configuration.
    - num_fold (int): Controls cross-validation. If None, the CV generator in the fold_strategy parameter of the setup function is used.
    - n_iter (int): Number of iterations in the grid search. Increasing 'n_iter' may improve model performance but also increases the training time.
    - custom_grid (any): To define custom search space for hyperparameters, pass a dictionary with parameter name and values to be iterated.
    
    ##### outPorts:
    - out_tuned_model (any): Tuned model object.
    """
    in_model: InArg[any]
    optimize: InArg[str]
    early_stopping_patience: InArg[int]
    num_fold: InArg[int]
    n_iter: InArg[int]
    custom_grid: InArg[any]
    out_tuned_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.optimize.value = "R2"
        self.num_fold.value = 10
        self.n_iter.value = 10

    def execute(self, ctx) -> None:
        from pycaret.regression import tune_model
        from IPython.display import display

        if self.early_stopping_patience.value is None:
            early_stopping = False
            patience = 10
        else:
            early_stopping = True
            patience = self.early_stopping_patience.value

        with capture.capture_output() as captured:
            tuned_model = tune_model(
                estimator=self.in_model.value,
                optimize=self.optimize.value,
                fold=self.num_fold.value,
                n_iter=self.n_iter.value,
                early_stopping=early_stopping,
                early_stopping_max_iters=patience,
                custom_grid=self.custom_grid.value
            )
        
        for o in captured.outputs:
            display(o)

        self.out_tuned_model.value = tuned_model

@xai_component(color="springgreen")
class PlotModelRegression(Component):
    """
    Analyzes the performance of a trained model.

    ##### inPorts:
    - in_model (any): Trained model object.
    - plot_type (str): Plot name.
    - list_available_plots (bool): List the available plots.
    
    ##### outPorts:
    - out_model (any): Trained model object.
    """
    in_model: InArg[any]
    plot_type: InArg[str]
    list_available_plots: InArg[bool]
    out_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.plot_type.value = 'residuals'
        self.list_available_plots.value = False

    def execute(self, ctx) -> None:
        from pycaret.regression import plot_model

        plot = {
            'residuals_interactive': 'Interactive Residual plots', 'residuals': 'Residuals Plot', 'error': 'Prediction Error Plot', 
            'cooks': 'Cooks Distance Plot', 'rfe': 'Recursive Feat. Selection', 'learning': 'Learning Curve', 'vc': 'Validation Curve',
            'manifold': 'Manifold Learning', 'feature': 'Feature Importance', 'feature_all': 'Feature Importance (All)', 
            'parameter': 'Model Hyperparameter', 'tree': 'Decision Tree'
        }
        
        with capture.capture_output() as captured:
            plot_model(self.in_model.value, plot=self.plot_type.value)
        captured.show()

        if self.list_available_plots.value:
            print('List of available plots (plot Type - Plot Name):')
            for key, value in plot.items():
                print(key, ' - ', value)

        self.out_model.value = self.in_model.value

@xai_component(color='crimson')
class FinalizeModelRegression(Component):
    """
    Trains a given estimator on the entire dataset including the holdout set.

    ##### inPorts:
    - in_model (any): Trained model object.
    
    ##### outPorts:
    - out_finalize_model (any): Trained model object.
    """
    in_model: InArg[any]
    out_finalize_model: OutArg[any]

    def execute(self, ctx) -> None:
        from pycaret.regression import finalize_model

        with capture.capture_output() as captured:
            finalized_model = finalize_model(self.in_model.value)
            print(finalized_model)
        captured.show()

        self.out_finalize_model.value = finalized_model

@xai_component(color='darkviolet')
class PredictModelRegression(Component):
    """
    Predicts Label and Score (probability of predicted class) using a trained model.
    When data is None, it predicts label and score on the holdout set.

    ##### inPorts:
    - in_model (any): Trained model object.
    - predict_dataset (any): Shape (n_samples, n_features). All features used during training must be available in the unseen dataset.
    
    ##### outPorts:
    - out_model (any): pandas.DataFrame with prediction and score columns.
    """
    in_model: InArg[any]
    predict_dataset: InArg[any]
    out_model: OutArg[any]

    def execute(self, ctx) -> None:
        from pycaret.regression import predict_model

        with capture.capture_output() as captured:
            prediction = predict_model(self.in_model.value, data=self.predict_dataset.value)
        captured.show()

        self.out_model.value = prediction

@xai_component(color='red')
class SaveModelRegression(Component):
    """
    Saves the transformation pipeline and trained model object into the current working directory as a pickle file for later use.

    ##### inPorts:
    - in_model (any): Trained model object.
    - save_path (str): Name and saving path of the model.
    - model_only (bool): When set to True, only the trained model object is saved instead of the entire pipeline.
    """
    in_model: InArg[any]
    save_path: InArg[str]
    model_only: InArg[bool]

    def execute(self, ctx) -> None:
        from pycaret.regression import save_model

        save_model(self.in_model.value, model_name=self.save_path.value, model_only=self.model_only.value)

@xai_component(color='red')
class LoadModelRegression(Component):
    """
    Loads a previously saved pipeline.

    ##### inPorts:
    - model_path (str): Name and path of the saved model.
    
    ##### outPorts:
    - model (any): Trained model object.
    """
    model_path: InArg[str]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        from pycaret.regression import load_model

        self.model.value = load_model(model_name=self.model_path.value)

@xai_component(color='gold')
class EnsembleModelRegression(Component):
    """
    Ensembles a given estimator. The output of this function is a score grid with CV scores by fold.

    ##### inPorts:
    - in_model (any): Trained model object.
    - method (str): Method for ensembling base estimator. It can be ‘Bagging’ or ‘Boosting’.
    - choose_better (bool): When set to True, the returned object is always better performing. The metric used for comparison is defined by the optimize parameter.
    - optimize (str): Metric to compare for model selection when choose_better is True.
    - n_estimators (int): The number of base estimators in the ensemble. In case of perfect fit, the learning procedure is stopped early.
    
    ##### outPorts:
    - out_ensemble_model (any): Trained model object.
    """
    in_model: InArg[any]
    method: InArg[str]
    choose_better: InArg[bool]
    optimize: InArg[str]
    n_estimators: InArg[int]
    out_ensemble_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.method.value = 'Bagging'
        self.choose_better.value = False
        self.optimize.value = 'R2'
        self.n_estimators.value = 10

    def execute(self, ctx) -> None:
        from pycaret.regression import ensemble_model

        with capture.capture_output() as captured:
            ensembled_model = ensemble_model(
                estimator=self.in_model.value,
                method=self.method.value,
                choose_better=self.choose_better.value,
                optimize=self.optimize.value,
                n_estimators=self.n_estimators.value
            )
        captured.show()
        print('Ensemble model:', ensembled_model)

        self.out_ensemble_model.value = ensembled_model

@xai_component(color='greenyellow')
class BlendModelsRegression(Component):
    """
    Trains a Soft Voting / Majority Rule classifier for select models passed in the top_model list.

    ##### inPorts:
    - top_models (any): List of trained model objects from CompareModel component.
    - model_1 (any): First model to blend.
    - model_2 (any): Second model to blend.
    - model_3 (any): Third model to blend.
    - choose_better (bool): When set to True, the returned object is always better performing. The metric used for comparison is defined by the optimize parameter.
    - optimize (str): Metric to compare for model selection when choose_better is True.
    
    ##### outPorts:
    - out_blended_model (any): Blended model object.
    """
    top_models: InArg[any]
    model_1: InArg[any]
    model_2: InArg[any]
    model_3: InArg[any]
    choose_better: InArg[bool]
    optimize: InArg[str]
    out_blended_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.choose_better.value = False
        self.optimize.value = 'R2'

    def execute(self, ctx) -> None:
        from pycaret.regression import blend_models

        model_list = self.top_models.value or [self.model_1.value, self.model_2.value, self.model_3.value]
        model_list = [model for model in model_list if model is not None]

        with capture.capture_output() as captured:
            blended_model = blend_models(estimator_list=model_list, choose_better=self.choose_better.value, optimize=self.optimize.value)
        captured.show()

        self.out_blended_model.value = blended_model

@xai_component(color='lawngreen')
class StackModelsRegression(Component):
    """
    Trains a meta model over select estimators passed in the estimator_list parameter.
    The output of this function is a score grid with CV scores by fold.

    ##### inPorts:
    - top_models (any): List of trained model objects from CompareModel component.
    - model_1 (any): First model to stack.
    - model_2 (any): Second model to stack.
    - model_3 (any): Third model to stack.
    - meta_model (any): When None, Logistic Regression is trained as a meta model.
    - choose_better (bool): When set to True, the returned object is always better performing. The metric used for comparison is defined by the optimize parameter.
    - optimize (str): Metric to compare for model selection when choose_better is True.
    
    ##### outPorts:
    - out_stacked_model (any): Trained model object.
    """
    top_models: InArg[any]
    model_1: InArg[any]
    model_2: InArg[any]
    model_3: InArg[any]
    meta_model: InArg[any]
    choose_better: InArg[bool]
    optimize: InArg[str]
    out_stacked_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.choose_better.value = False
        self.optimize.value = 'R2'

    def execute(self, ctx) -> None:
        from pycaret.regression import stack_models

        model_list = self.top_models.value or [self.model_1.value, self.model_2.value, self.model_3.value]
        model_list = [model for model in model_list if model is not None]

        with capture.capture_output() as captured:
            stacked_model = stack_models(
                estimator_list=model_list,
                meta_model=self.meta_model.value,
                choose_better=self.choose_better.value,
                optimize=self.optimize.value
            )
        captured.show()
        
        print('Stacked models:', stacked_model.estimators_)

        self.out_stacked_model.value = stacked_model

@xai_component(color="yellow")
class AutoMLRegression(Component):
    """
    Returns the best model out of all trained models in the current session based on the optimize parameter.
    Metrics evaluated can be accessed using the get_metrics function.

    ##### inPorts:
    - optimize (str): Metric to use for model selection. It also accepts custom metrics added using the add_metric function.
    
    ##### outPorts:
    - best_model (any): Best trained model object.
    """
    optimize: InArg[str]
    best_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.optimize.value = 'R2'

    def execute(self, ctx) -> None:
        from pycaret.regression import automl

        self.best_model.value = automl(optimize=self.optimize.value)
