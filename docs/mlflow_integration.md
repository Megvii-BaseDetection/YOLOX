## MLFlow Integration
YOLOX now supports MLFlow integration. MLFlow is an open-source platform for managing the end-to-end machine learning lifecycle. It is designed to work with any ML library, algorithm, deployment tool, or language. MLFlow can be used to track experiments, metrics, and parameters, and to log and visualize model artifacts. \
For more information, please refer to: [MLFlow Documentation](https://www.mlflow.org/docs/latest/index.html)

## Follow these steps to start logging your experiments to MLFlow:
### Step-1: Install MLFlow via pip 
```bash
pip install mlflow python-dotenv
```

### Step-2: Set up MLFlow Tracking Server
Start or connect to a MLFlow tracking server like databricks. You can start a local tracking server by running the following command:
```bash
mlflow server --host 127.0.0.1 --port 8080
```
Read more about setting up MLFlow tracking server [here](https://mlflow.org/docs/latest/tracking/server.html#mlflow-tracking-server)

### Step-3: Set up MLFlow Environment Variables
Set the following environment variables in your `.env` file:
```bash
MLFLOW_TRACKING_URI="127.0.0.1:5000"  # set to your mlflow server URI
MLFLOW_EXPERIMENT_NAME="/path/to/experiment"  # set to your experiment name
MLFLOW_TAGS={"release.candidate": "DEV1", "release.version": "0.0.0"}
# config related to logging model to mlflow as pyfunc
YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS="True" # whether to log model (best or historical) or not 
YOLOX_MLFLOW_LOG_MODEL_PER_n_EPOCHS=30 # try logging model only after every n epochs
YOLOX_MLFLOW_LOG_Nth_EPOCH_MODELS="False" # whether to log step model along with best_model or not
YOLOX_MLFLOW_RUN_NAME="" # give a custom name to your run, otherwise a random name is assign by mlflow
YOLOX_MLFLOW_FLATTEN_PARAMS="True" # flatten any sub sub params of dict to be logged as simple key value pair


MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=True # log system gpu usage and other metrices
MLFLOW_NESTED_RUN="False" #whether to run as a nested run of given run_id
MLFLOW_RUN_ID="" # continue training from a given run_id
```
### Step-5: Provide --logger "mlflow" to the training script
```bash
python tools/train.py -l mlflow -f exps/path/to/exp.py -d 1 -b 8 --fp16 -o -c 
pre_trained_model/<model>.pth
# note the -l mlflow flag
# one working example is this
python tools/train.py -l mlflow -f exps/example/custom/yolox_s.py -d 1 -b 8 --fp16 -o -c pre_trained_model/yolox_s.pth
```
### Step-4: optional; start the mlflow ui and track your experiments
If you log runs to a local mlruns directory, run the following command in the directory above it, then access http://127.0.0.1:5000 in your browser.

```bash
mlflow ui --port 5000
```

## Optional Databricks Integration

### Step-1: Install Databricks sdk
```bash
pip install databricks-sdk
```

### Step-2: Set up Databricks Environment Variables
Set the following environment variables in your `.env` file:
```bash
MLFLOW_TRACKING_URI="databricks"  # set to databricks
MLFLOW_EXPERIMENT_NAME="/Users/<user>/<experiment_name>/"
DATABRICKS_HOST = "https://dbc-1234567890123456.cloud.databricks.com" # set to your server URI
DATABRICKS_TOKEN = "dapixxxxxxxxxxxxx"
```