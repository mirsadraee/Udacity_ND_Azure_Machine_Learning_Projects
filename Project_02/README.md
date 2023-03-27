# Operationalizing Machine Learning

## Architectural Diagram
An architectural diagram of operations for operationalizing machine learning is shown in Figure 1.

<img src="figures/Flowchart.png"  width="600">

Figure 1: Operationalizing Machine Learning (Image is taken from: Udacity MLEMA Nanodegree Course)

## Key Steps
The key steps are section of this project:
1. Authentication
2. Automated ML Experiment
3. Deploy the best model
4. Enable logging
5. Swagger Documentation
6. Consume model endpoints
7. Create and publish a pipeline

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
### 1. Authentication
This step is jumped over as it could not be implemented due to insufficient authorization for creating a security principal.

### 2. Automated ML Experiment
#### Take a screenshot of “Registered Datasets” in ML Studio showing that Bankmarketing dataset available
In this project the data of xxx are examined, a machine learning model was made, trainedand etc.

<img src="figures/data_header.png"  width="600">

The data can be analyzed in advanced and check some metrics before starting the experiment.

<img src="figures/data_analysis.png"  width="600">


#### Take a screenshot showing that the experiment is shown as completed
The experiment started successfully and took around 1 hour to be completed, the maximum training time was set to 1 hour so the resources would not be overloaded by model training.

<img src="figures/experiment_started.png"  width="600">

The experiment was completed successfully and variety of models were trained using AutoML.

<img src="figures/experiment_completed_status.png"  width="600">

A short summary of the job is given below:

<img src="figures/experiment_completed.png"  width="600">

A section of trained models is given below:

<img src="figures/experiment_results.png"  width="600">

#### Take a screenshot of the best model after the experiment completes
1. The best model for deployment is the one from VotingEnsemble training which can predict the target value by 95%. 

<img src="figures/experiment_best_model.png"  width="600">

The best model metrics are given in figure below:

<img src="figures/experiment_best_model_metrics.png"  width="600">

A short explanation of the best model is given bwelow and showing the important of input variables with highest impact:

<img src="figures/experiment_best_model_explanation.png"  width="600">

The AutoML Model Training was done successfully, so the best model is ready for deployment.

### 3. Deploy the best model
1. Select the best model for deployment

2. Deploy the model and enable "Authentication"

#### Select the best model for deployment
#### Deploy the model and enable "Authentication"
#### Deploy the model using Azure Container Instance (ACI)


### 4. Enable logging
1. Please complete the following using Azure Python SDK:
2. Ensure az is installed, as well as the Python SDK for Azure
3. Create a new virtual environment with Python3
4. Write and run code to enable Application Insights
5. Use the provided code logs.py to view the logs

#### Take a screenshot showing that "Application Insights" is enabled in the Details tab of the endpoint.
#### Take a screenshot showing logs by running the provided logs.py script

### 5. Swagger Documentation
1. Download the swagger.json file
2. Run the swagger.sh and serve.py
3. Interact with the swagger instance running with the documentation for the HTTP API of the model.
4. Display the contents of the API for the model

#### Take a screenshot showing that swagger runs on localhost showing the HTTP API methods and responses for the model

### 6. Consume model endpoints
1. Modifying both the scoring_uri and the key to match the key for your service and the URI that was generated after deployment
2.  Execute the endpoint.py file, the output should be similar to the following: {"result": ["yes", "no"]}

#### Take a screenshot showing that theendpoint.py script runs against the API producing JSON output from the model.

### 7. Create and publish a pipeline
1. Run through all the cells in the provided Notebook which uses the Python SDK.
2. Upload the Jupyter Notebook aml-pipelines-with-automated-machine-learning-step.ipynb to the Azure ML studio
3. Update all the variables that are noted to match your environment
4. Make sure a config.json has been downloaded and is available in the current working directory
5. Run through the cells
6. Verify the pipeline has been created and shows in Azure ML studio, in the Pipelines section
7. Verify that the pipeline has been scheduled to run or is running
Task List

Again:
1. The pipeline section of Azure ML studio, showing that the pipeline has been created
2. The pipelines section in Azure ML Studio, showing the Pipeline Endpoint
3. The Bankmarketing dataset with the AutoML module
4. The “Published Pipeline overview”, showing a REST endpoint and a status of ACTIVE
5. In Jupyter Notebook, showing that the “Use RunDetails Widget” shows the step runs
6. In ML studio showing the scheduled run

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
