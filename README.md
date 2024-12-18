<p align="center">
  <a href="https://github.com/XpressAI/xircuits/tree/master/xai_components#xircuits-component-library-list">Component Libraries</a> •
  <a href="https://github.com/XpressAI/xircuits/tree/master/project-templates#xircuits-project-templates-list">Project Templates</a>
  <br>
  <a href="https://xircuits.io/">Docs</a> •
  <a href="https://xircuits.io/docs/Installation">Install</a> •
  <a href="https://xircuits.io/docs/category/tutorials">Tutorials</a> •
  <a href="https://xircuits.io/docs/category/developer-guide">Developer Guides</a> •
  <a href="https://github.com/XpressAI/xircuits/blob/master/CONTRIBUTING.md">Contribute</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
</p>





<p align="center"><i>Xircuits Component Library to interface with 
Anthropic AI! Build AI-powered solutions.</i></p>

---
## Xircuits Component Library for PyCaret

This library consists of PyCaret components that allow implementing of the different AutoML tasks in Xircuits:

1. Classification
2. Regression 
3. Anomaly Detection 
4. Clustering 
5. Natural Language Processing (NLP)

We have supported this library with multiple Xircuits implantations for most of the AutoML tasks, these can be found in the `examples/` folder.

## Table of Contents

- [Preview](#preview)
- [Prerequisites](#prerequisites)
- [Main Xircuits Components](#main-xircuits-components)
- [Try the Examples](#try-the-examples)
- [Installation](#installation)

## Preview

### AutoMLBasicAnomalyDetection Example:

![AutoMLBasicAnomalyDetection](https://github.com/user-attachments/assets/b4b6b764-9afa-402d-aecf-b1e9ce586ee7)

### AutoMLBasicAnomalyDetection Result:

![AutoMLBasicAnomalyDetection-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/ed5e1bc7-c96c-4c2d-bdad-0d7117090928)

## Prerequisites

Before you begin, you will need the following:

1. Python3.9+.
2. Xircuits.

## Main Xircuits Components

### SetupClassification Component:
Initializes the training environment and creates a transformation pipeline for classification tasks. It prepares data for modeling by applying preprocessing, normalization, and other transformations.


<img src="https://github.com/user-attachments/assets/4f3edb8f-d07c-422b-9a93-3a5564b93625" alt="SetupClassification" width="200" height="300" />

### CompareModelsClassification Component:
Compares all available classification models and ranks them based on cross-validation scores. Outputs the top model(s) for further use.

<img src="https://github.com/user-attachments/assets/54ce5b00-28b3-40b1-8cd9-6b352b170be1" alt="CompareModelsClassification" width="200" height="100" />

### CreateModelClassification Component:
Trains a specified classification model and evaluates its performance using cross-validation. Outputs a trained model object.

### TuneModelClassification Component:
Tunes the hyperparameters of a classification model to optimize a specified metric. Outputs a tuned model with the best parameter configuration.

### PlotModelClassification Component:
Generates performance visualizations for a trained classification model, such as feature importance, confusion matrix, and ROC curve.

### PredictModelClassification Component:
Generates predictions for a new dataset using a trained classification model. Outputs a DataFrame containing predictions and scores.

### FinalizeModelClassification Component:
Finalizes a classification model by training it on the entire dataset, including the holdout set, for production deployment.

### AutoMLClassification Component:
Automatically selects the best classification model based on a specified metric from all trained models in the current session.

### SaveModelClassification Component:
Saves a trained classification model to a file for future use. Allows saving either the entire pipeline or just the model.

### LoadModelClassification Component:
Loads a previously saved classification model pipeline for reuse in predictions or further analysis.

## Try The Examples

We have provided an example workflow to help you get started with the PyCaret component library. Give it a try and see how you can create custom PyCaret components for your applications.

### AutoML Basic Anomaly Detection Example  
Check out the AutoMLBasicAnomalyDetection.xircuits workflow. This example uses PyCaret's anomaly detection components to set up, train, and evaluate an Isolation Forest (iForest) model on the mice dataset. Key steps include dataset preparation, model creation, assigning anomaly labels, and visualizing the results with plots. The trained anomaly detection model is saved for future use.



### AutoML Basic Binary Classification Example  
Check out the AutoMLBasicBinaryClassification.xircuits workflow. This example demonstrates a binary classification pipeline using PyCaret. The workflow processes the credit dataset, identifies the best model based on F1 score, and tunes it. Plots such as feature importance, precision-recall, and confusion matrix are generated to evaluate the model. Finally, the trained model is saved for deployment.



### AutoML Basic Clustering Example  
Check out the AutoMLBasicClustering.xircuits workflow. This example demonstrates a clustering pipeline using PyCaret components. The workflow processes the mice dataset to perform clustering with a k-means algorithm. Key steps include dataset setup, clustering model creation, and evaluation through various plots, such as cluster visualization, elbow plot, silhouette plot, and feature distribution. The trained clustering model is saved for future use.



### AutoML Basic Multiclass Classification Example  
Check out the AutoMLBasicMulticlassClassification.xircuits workflow. This example illustrates a multiclass classification pipeline using PyCaret components. The workflow processes the iris dataset, identifies the best-performing logistic regression model, and evaluates its performance using plots like confusion matrix, classification report, decision boundary, and error analysis. The finalized model is saved as the best-trained model for future predictions.



### AutoML Basic Regression Example  
Check out the AutoMLBasicRegression.xircuits workflow. This example demonstrates a regression pipeline using PyCaret components. The workflow uses the diamond dataset to predict prices by training an AdaBoost model. The model is tuned for mean absolute error (MAE), visualized with available plots, and finalized for deployment. The trained regression model is saved for reuse.



### AutoML Classification Blend Models Example  
Check out the AutoMLClassificationBlendModels.xircuits workflow. This example demonstrates blending top-performing models in a binary classification pipeline using PyCaret. The credit dataset is processed to create a "soft voting" ensemble of the top three models optimized for the F1 score. The workflow includes calibration and validation using calibration plots to improve the ensemble model's predictions.


### AutoML Regression Stack Models Example  
Check out the AutoMLRegressionStackModels.xircuits workflow. This example showcases stacking models in a regression pipeline using PyCaret. The workflow processes the diamond dataset, identifies the top three models, and stacks them with XGBoost as a meta-model. The stacked model's performance is evaluated with parameter and Cook's distance plots. Finally, the stacked regression model is saved for deployment.

## Installation
To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the PyCaret library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:

```
xircuits install pycaret
```
You can also do it manually by cloning and installing it:
```
# base Xircuits directory
git clone https://github.com/XpressAI/xai-pycaret xai_components/xai_pycaretc
pip install -r xai_components/xai_pycaret/requirements.txt 