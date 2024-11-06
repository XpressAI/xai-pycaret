from xai_components.base import InArg, OutArg, Component, xai_component
from IPython.utils import capture

@xai_component(color="blue")
class SetupClustering(Component):
    """
    Initializes the training environment and creates the transformation pipeline.
    Setup must be called before executing any other function.

    ##### inPorts:
    - in_dataset (any): Shape (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.
    - preprocess (bool): When set to False, no transformations are applied. Data must be ready for modeling (no missing values, no dates, categorical data encoding).
    - normalize (bool): When set to True, it transforms the numeric features by scaling them to a given range.
    - transformation (bool): When set to True, it applies the power transform to make data more Gaussian-like.
    - remove_multicollinearity (bool): When set to True, features with inter-correlations higher than the defined threshold are removed.
    - multicollinearity_threshold (float): Threshold for correlated features. Ignored when remove_multicollinearity is not True.
    - bin_numeric_features (any): To convert numeric features into categorical. It takes a list of strings with column names that are related.
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
    remove_multicollinearity: InArg[bool]
    multicollinearity_threshold: InArg[float]
    bin_numeric_features: InArg[any]
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
        self.remove_multicollinearity.value = False
        self.multicollinearity_threshold.value = 0.9
        self.log_experiment.value = False

        self.experiment_name.value = 'default'
        self.use_gpu.value = False

    def execute(self, ctx) -> None:
        from pycaret.clustering import setup, models

        if self.seed.value is None:
            print("Set the seed value for reproducibility.")
            
        with capture.capture_output() as captured:
            setup_pycaret = setup(
                data=self.in_dataset.value,
                preprocess=self.preprocess.value,
                normalize=self.normalize.value,
                transformation=self.transformation.value,
                remove_multicollinearity=self.remove_multicollinearity.value,
                multicollinearity_threshold=self.multicollinearity_threshold.value,
                bin_numeric_features=self.bin_numeric_features.value,
                ignore_features=self.ignore_features.value,
                session_id=self.seed.value,
                log_experiment=self.log_experiment.value,
                experiment_name=self.experiment_name.value,
                use_gpu=self.use_gpu.value
            )

        captured.show()
        
        print("List of the Available Clustering models: ")
        print(models())

@xai_component(color="orange")
class CreateModelClustering(Component):
    """
    Trains a given model from the model library. All available models can be accessed using the models function.

    ##### inPorts:
    - model_id (str): ID of a model available in the model library or pass an untrained model object consistent with scikit-learn API.
    - num_clusters (int): The number of clusters to form.
    
    ##### outPorts:
    - out_created_model (any): Trained Model object.
    """
    model_id: InArg[str]
    num_clusters: InArg[int]
    out_created_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.model_id.value = 'knn'
        self.num_clusters.value = 4

    def execute(self, ctx) -> None:
        from pycaret.clustering import create_model

        with capture.capture_output() as captured:
            created_model = create_model(
                model=self.model_id.value,
                num_clusters=self.num_clusters.value
            )
        captured.show()
        print(created_model)

        self.out_created_model.value = created_model

@xai_component(color="salmon")
class TuneModelAnomaly(Component):
    """
    Tunes the hyperparameters of a given model. The output of this component is a score grid with CV scores by fold of the best selected model based on optimize parameter.

    ##### inPorts:
    - model_id (str): Trained model object.
    - supervised_target (str): Name of the target column containing labels.
    - supervised_type (str): Type of task. ‘classification’ or ‘regression’. Automatically inferred when None.
    - supervised_estimator (str): The classification or regression model.
    - optimize (str): The classification or regression optimizer.
    - custom_grid (any): To define custom search space for hyperparameters, pass a dictionary with parameter name and values to be iterated.
    
    ##### outPorts:
    - out_tuned_model (any): Tuned model object.
    """
    model_id: InArg[str]
    supervised_target: InArg[str]
    supervised_type: InArg[str]
    supervised_estimator: InArg[str]
    optimize: InArg[str]
    custom_grid: InArg[any]
    out_tuned_model: OutArg[any]

    def execute(self, ctx) -> None:
        from pycaret.anomaly import tune_model
        from IPython.display import display

        with capture.capture_output() as captured:
            tuned_model = tune_model(
                model=self.model_id.value,
                supervised_target=self.supervised_target.value,
                supervised_type=self.supervised_type.value,
                supervised_estimator=self.supervised_estimator.value,
                optimize=self.optimize.value,
                custom_grid=self.custom_grid.value
            )
        captured.show()

        self.out_tuned_model.value = tuned_model

@xai_component(color="firebrick")
class AssignModelClustering(Component):
    """
    Assigns cluster labels to the dataset for a given model.

    ##### inPorts:
    - in_model (any): Trained Model Object.
    
    ##### outPorts:
    - out_model (any): Trained Model Object with assigned labels.
    """
    in_model: InArg[any]
    out_model: OutArg[any]

    def execute(self, ctx) -> None:
        from pycaret.clustering import assign_model

        with capture.capture_output() as captured:
            assigned_model = assign_model(model=self.in_model.value)
        captured.show()
        print(assigned_model.head())

        self.out_model.value = self.in_model.value

@xai_component(color='darkviolet')
class PredictModelClustering(Component):
    """
    Generates cluster labels using a trained model.

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
        from pycaret.clustering import predict_model

        with capture.capture_output() as captured:
            prediction = predict_model(self.in_model.value, data=self.predict_dataset.value)
        captured.show()
        print(prediction.head())

        self.out_model.value = prediction

@xai_component(color="springgreen")
class PlotModelClustering(Component):
    """
    Analyzes the performance of a trained model.

    ##### inPorts:
    - in_model (any): Trained model object.
    - plot_type (str): Plot name.
    - feature (str): Feature to be evaluated when plot = ‘distribution’. When plot type is ‘cluster’ or ‘tsne’ feature column is used as a hoverover tooltip and/or label when the label param is set to True. When the plot type is ‘cluster’ or ‘tsne’ and feature is None, first column of the dataset is used.
    - list_available_plots (bool): List the available plots.
    
    ##### outPorts:
    - out_model (any): Trained model object.
    """
    in_model: InArg[any]
    plot_type: InArg[str]
    feature: InArg[str]
    list_available_plots: InArg[bool]
    out_model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.plot_type.value = 'cluster'
        self.list_available_plots.value = False

    def execute(self, ctx) -> None:
        from pycaret.clustering import plot_model

        plot = {
            'cluster': 'Cluster PCA Plot (2d)',
            'tsne': 'Cluster TSnE (3d)',
            'elbow': 'Elbow Plot',
            'silhouette': 'Silhouette Plot',
            'distance': 'Distance Plot',
            'distribution': 'Distribution Plot'
        }

        with capture.capture_output() as captured:
            plot_model(self.in_model.value, plot=self.plot_type.value, feature=self.feature.value)
        captured.show()

        if self.list_available_plots.value:
            print('List of available plots (plot Type - Plot Name):')
            for key, value in plot.items():
                print(key, ' - ', value)

        self.out_model.value = self.in_model.value

@xai_component(color='red')
class SaveModelClustering(Component):
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
        from pycaret.clustering import save_model

        save_model(self.in_model.value, model_name=self.save_path.value, model_only=self.model_only.value)

@xai_component(color='red')
class LoadModelClustering(Component):
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
        from pycaret.clustering import load_model

        self.model.value = load_model(model_name=self.model_path.value)
