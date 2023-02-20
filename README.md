# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Your project description here.

## Setup

Create a conda environment using the ```environment.yml file```
```
conda env create -f environment.yml
```

I recommend using VS Code and the Python extensions. 
- Linting can be configured
- Document formatting can be configured
- Easy running of python files
- Testing integrated

## Files and data description
### Files
- churn_pipeline.py: Runnable pipeline that performes data transformation and modelling
- environment.yml: Environment file (see above)

### Udacity files (not needed for the module itself)
- churn_notebook.ipynb: Original notebook provided by Udacity
- Guide.ipynb: Original Guide provided by Udacity
- churn_library.py: wrapper around the churn module that delegates the calles to the module
- churn_script_logging_and_tests.py: an integration test (similar to the pipeline) with logging exercise

### Folders
- churn: Contains the main module where all the logic is encapsulated (into several classes). Tests can also be found here.
- data: Data for the project
- docs: outputs of data transformation and modelling (mainly images)
- logs: folder with application logs
- model: folder with pkl files of the best models found

## Running Files
The pipeline and all related steps (data transformation, model training and evaluation) can be started by
- activating the conda environment ```conda activate predict-customer-churn```
- and by executing the script ```churn_pipeline.py``` (alternativley ```churn_script_logging_and_tests.py```)

Alternatively it can be executed by running the script 

## Checking files and code quality
Navigate to the root folder of the project
```
pylint ./churn
pylint pylint churn_pipeline.py
pylint churn_script_logging_and_tests.py
pylint churn_library.py
```
Every file is currently >9.

## Executing test
Navigate to the root folder of the project to execute the unit tests
```
pytest
```


