from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, dynalist, dynatuple
from IPython.utils import capture

@xai_component(color="blue")
class SetupAnomaly(Component):
    """
    Initializes the training environment and creates the transformation pipeline.
    Setup must be called before executing any other function.

    ##### inPorts:
    - in_dataset (any): Shape (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.
    - preprocess (bool): When set to False, no transformations are applied. Data must be ready for modeling (no missing values, no dates, categorical data encoding).
    - normalize (bool): When set to True, it transforms the numeric features by scaling them to a given range.
    - transformation (bool): When set to True, it applies the power transform to make data more Gaussian-like.
    - ignore_low_variance (bool): When set to True, all categorical features with insignificant variances are removed from the data.
    - remove_multicollinearity (bool): When set to True, features with inter-correlations higher than the defined threshold are removed.
    - multicollinearity_threshold (float): Threshold for correlated features. Ignored when remove_multicollinearity is not True.
    - combine_rare_levels (bool): When set to True, frequency percentile for levels in categorical features below a certain threshold is combined into a single level.
    - rare_level_threshold (float): Percentile distribution below which rare categories are combined. Ignored when combine_rare_levels is not True.
    - bin_numeric_features (any): To convert numeric features into categorical. It takes a list of strings with column names that are related.
    - group_features (any): When the dataset contains features with related characteristics, group_features parameter can be used for feature extraction. It takes a list of strings with column names that are related.
    - ignore_features (list): Ignore_features param can be used to ignore features during model training. It takes a list of strings with column names that are to be ignored.
    - seed (int): You can use random_state for reproducibility.
    - log_experiment (bool): Logging setup and training.
    - experiment_name (str): Name of the experiment for logging.
    - use_gpu (bool): Whether to use GPU for training.
    """
    in_dataset: InArg[any]
    preprocess: InArg[bool]
    normalize: InArg[bool]
    transformation: InArg[bool]
    ignore_low_variance: InArg[bool]
    remove_multicollinearity: InArg[bool]
    multicollinearity_threshold: InArg[float]
    combine_rare_levels: InArg[bool]
    rare_level_threshold: InArg[float]
    bin_numeric_features: InArg[any]
    group_features: InArg[any]
    ignore_features: InArg[list]
    seed: InArg[int]
    log_experiment: InArg[bool]
    experiment_name: InArg[str]
    use_gpu: InArg[bool]

    def __init__(self):
        super().__init__()
        self.preprocess.value = True
        self.normalize.value = False
        self.transformation.value = False
        self.ignore_low_variance.value = False
        self.remove_multicollinearity.value = False
        self.multicollinearity_threshold.value = 0.9
        self.combine_rare_levels.value = False
        self.rare_level_threshold.value = 0.1
        self.experiment_name.value = 'default'
        self.use_gpu.value = False

    def execute(self, ctx) -> None:
        from pycaret.anomaly import setup, models

        in_dataset = self.in_dataset.value
        preprocess = self.preprocess.value
        normalize = self.normalize.value
        transformation = self.transformation.value
        remove_multicollinearity = self.remove_multicollinearity.value
        multicollinearity_threshold = self.multicollinearity_threshold.value
        bin_numeric_features = self.bin_numeric_features.value
        group_features = self.group_features.value
        ignore_features = self.ignore_features.value
        seed = self.seed.value
        log_experiment = self.log_experiment.value
        experiment_name = self.experiment_name.value
        use_gpu = self.use_gpu.value

        if seed is None:
            print("Set the seed value for reproducibility.")
            
        with capture.capture_output() as captured:
            setup_pycaret = setup(data=in_dataset,
                                  preprocess=preprocess,
                                  normalize=normalize,
                                  transformation=transformation,
                                  remove_multicollinearity=remove_multicollinearity,
                                  multicollinearity_threshold=multicollinearity_threshold,
                                  bin_numeric_features=bin_numeric_features,
                                  group_features=group_features,
                                  ignore_features=ignore_features,
                                  session_id=seed,
                                  log_experiment=log_experiment,
                                  experiment_name=experiment_name,
                                  use_gpu=use_gpu)

        captured.show()

        print("List of the Available Anomaly Detection models: ")
        print(models())

@xai_component(color="orange")
class CreateModelAnomaly(Component):
    """
    Trains a given model from the model library. All available models can be accessed using the models function.

    ##### inPorts:
    - model_id (str): ID of a model available in the model library or pass an untrained model object consistent with scikit-learn API.
    - fraction (float): The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
    
    ##### outPorts:
    - out_created_model (any): Trained Model object.
    """
    model_id: InArg[str]
    fraction: InArg[float]
    out_created_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.model_id.value = 'knn'
        self.fraction.value = 0.05

    def execute(self, ctx) -> None:
        from pycaret.anomaly import create_model

        model_id = self.model_id.value
        fraction = self.fraction.value

        with capture.capture_output() as captured:
            created_model = create_model(model=model_id, fraction=fraction)
        captured.show()
        print(created_model)

        self.out_created_model.value = created_model

@xai_component(color="salmon")
class TuneModelAnomaly(Component):
    """
    Tunes the hyperparameters of a given model. The output of this component is a score grid with CV scores by fold of the best selected model based on the optimize parameter.

    ##### inPorts:
    - model_id (str): Trained model object.
    - supervised_target (str): Name of the target column containing labels.
    - supervised_type (str): Type of task. ‘classification’ or ‘regression’. Automatically inferred when None.
    - supervised_estimator (str): The classification or regression model.
    - method (str): Method to handle anomalies.
    - optimize (str): The classification or regression optimizer.
    - custom_grid (any): To define a custom search space for hyperparameters, pass a dictionary with parameter name and values to be iterated.
    
    ##### outPorts:
    - out_tuned_model (any): Tuned model object.
    """
    model_id: InArg[str]
    supervised_target: InArg[str]
    supervised_type: InArg[str]
    supervised_estimator: InArg[str]
    method: InArg[str]
    optimize: InArg[str]
    custom_grid: InArg[any]
    out_tuned_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.method.value = 'drop'

    def execute(self, ctx) -> None:
        from pycaret.anomaly import tune_model

        model_id = self.model_id.value
        supervised_target = self.supervised_target.value
        supervised_type = self.supervised_type.value
        supervised_estimator = self.supervised_estimator.value
        method = self.method.value
        optimize = self.optimize.value
        custom_grid = self.custom_grid.value

        with capture.capture_output() as captured:
            tuned_model = tune_model(model=model_id,
                                     supervised_target=supervised_target,
                                     supervised_type=supervised_type,
                                     supervised_estimator=supervised_estimator,
                                     method=method,
                                     optimize=optimize,
                                     custom_grid=custom_grid)
        captured.show()

        self.out_tuned_model.value = tuned_model

@xai_component(color="firebrick")
class AssignModelAnomaly(Component):
    """
    Assigns anomaly labels to the dataset for a given model (1 = outlier, 0 = inlier).

    ##### inPorts:
    - in_model (any): Trained Model Object.
    
    ##### outPorts:
    - out_model (any): Trained Model Object with assigned labels.
    """
    in_model: InArg[any]
    out_model: OutArg[any]

    def execute(self, ctx) -> None:
        from pycaret.anomaly import assign_model

        in_model = self.in_model.value

        with capture.capture_output() as captured:
            assigned_model = assign_model(model=in_model)
        captured.show()
        print(assigned_model.head())

        self.out_model.value = in_model

@xai_component(color='darkviolet')
class PredictModelAnomaly(Component):
    """
    Generates anomaly labels using a trained model.

    ##### inPorts:
    - in_model (any): Trained model object.
    - predict_dataset (any): Shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    ##### outPorts:
    - out_model (any): pandas.DataFrame with prediction and score columns.
    """
    in_model: InArg[any]
    predict_dataset: InArg[any]
    out_model: OutArg[any]

    def execute(self, ctx) -> None:
        from pycaret.anomaly import predict_model

        in_model = self.in_model.value
        predict_dataset = self.predict_dataset.value

        with capture.capture_output() as captured:
            prediction = predict_model(in_model, data=predict_dataset)
        captured.show()
        print(prediction.head())

        self.out_model.value = prediction

@xai_component(color="springgreen")
class PlotModelAnomaly(Component):
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
        self.plot_type.value = 'tsne'
        self.list_available_plots.value = False

    def execute(self, ctx) -> None:
        from pycaret.anomaly import plot_model

        plot = {'tsne': 't-SNE (3d) Dimension Plot', 'umap': 'UMAP Dimensionality Plot'}
        
        in_model = self.in_model.value
        plot_type = self.plot_type.value
        list_available_plots = self.list_available_plots.value

        with capture.capture_output() as captured:
            plot_model(in_model, plot=plot_type)
        captured.show()

        if list_available_plots:
            print('List of available plots (plot Type - Plot Name):')
            for key, value in plot.items():
                print(key, ' - ', value)

        self.out_model.value = in_model

@xai_component(color='red')
class SaveModelAnomaly(Component):
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
        from pycaret.anomaly import save_model

        in_model = self.in_model.value
        save_path = self.save_path.value
        model_only = self.model_only.value

        save_model(in_model, model_name=save_path, model_only=model_only)

@xai_component(color='red')
class LoadModelAnomaly(Component):
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
        from pycaret.anomaly import load_model

        model_path = self.model_path.value
        loaded_model = load_model(model_name=model_path)
        
        self.model.value = loaded_model
