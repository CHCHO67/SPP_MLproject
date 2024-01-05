# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout

import mlflow
import mlflow.keras
import mlflow.server

from os import path
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

from mlflow.tracking import MlflowClient

yf.pdr_override()

# %%
# 아래 코드는 각 기업 데이터 수집하고 이를 하나의 데이터 프레임으로 합쳐서 분석할 수 있또록 준비하는 것
# Set up End and Start times for data grab
"""
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
    # stock에 대한 전역 변수 생성
    globals()[stock] = yf.download(stock, start, end)
    print(type(globals()[stock]))
    print(globals()[stock])

company_list = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

df = pd.concat(company_list, axis=0)
"""
# df.to_csv('concat_test.csv', index=False)
# df.shape
# df.head(10)
# df.tail(10)


# company.head(100)
# %%
def make_Dataset(stock_code="AAPL", start_data="2012-01-01", end_date=datetime.now):
    # Define the file path
    file_path = f"{stock_code}_data.csv"

    # Check if the file exists
    if os.path.isfile(file_path):
        # Read the dataframe from the file
        df = pd.read_csv(file_path)
    else:
        # Get data from yf
        df = pdr.get_data_yahoo(stock_code, start=start_data, end=end_date)
        # Save the dataframe to a file
        df.to_csv(file_path)

    # Filter 'Close' column
    data = df.filter(["Close"])
    dataset = data.values

    # Set training data length
    training_data_len = int(np.ceil(len(dataset) * 0.95))
    training_data_len

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    return scaler, scaled_data, training_data_len, dataset, data

def prepare_training_data(scaled_data, training_data_len):
    # Create the training data set
    train_data = scaled_data[0 : int(training_data_len), :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #
    x_train.shape
    return x_train, y_train

def build_improved_LSTM_model_with_MLflow(
    x_train, y_train, batch_size=1, epochs=1, model_name="LSTM_model"
):

    # mlflow web ui 설정
    mlflow.set_tracking_uri("http://192.168.200.0:80")
    client = MlflowClient(tracking_uri="http://192.168.200.0:80")
    
    # 원하는 실험 이름
    experiment_name = "new test"  
    
    # 새로운 Experiment 생성 혹은 이미 존재하는 Experiment에 접근
    if client.get_experiment_by_name(experiment_name) is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    print(f"Experiment ID: {experiment_id}")
    
    run = client.create_run(experiment_id)
    # Search for existing runs
# 최근 실행 중 2개만 검색
    existing_runs = client.search_runs(experiment_ids=[experiment_id], max_results=2, order_by=["attribute.start_time DESC"])
    print(f"Existing runs (last 2): {existing_runs}")

    #Determine the next run number
    run_numbers = [
        int(run.data.tags.get('mlflow.runName', '').split("#")[-1])
        for run in existing_runs
        if run.data.tags.get('mlflow.runName', '').startswith(experiment_name)
    ]

    next_run_number = max(run_numbers) + 1 if run_numbers else 1

    run_name=f"{experiment_name} #{next_run_number}"
    client.set_tag(run.info.run_id, 'mlflow.runName', run_name)
    print(f"run name: {run_name}")
    # 새로운 Run 실행, run_name 설정
    #print(f"run: {run.info.run_id}")
    with mlflow.start_run(run_id=run.info.run_id) as run:
        # Access current run information
        run_name = run.data.tags.get('mlflow.runName')
        print(run_name)
        #현재 설정된 Experiment 및 Run 정보 확인
        current_experiment = client.get_experiment(run.info.experiment_id)
        current_run = client.get_run(run.info.run_id)

        print(f"현재 설정된 Experiment: {current_experiment.name}")
        print(f"현재 실행 중인 Run: {current_run}")

        # LSTM Model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        # model.add(Dropout(0.2))  # Adding dropout for regularization
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        # model.add(Dropout(0.2))  # Adding dropout for regularization
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer="adam", loss="mean_squared_error")
        
        # Train the model
              # Train the model
        with tf.device("/GPU:0"):
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        # Log parameters
        client.log_param(run.info.run_id, "batch_size", batch_size)
        client.log_param(run.info.run_id, "epochs", epochs)

        # Log metrics
        loss_values = history.history["loss"]
        for epoch, loss in enumerate(loss_values):
            client.log_metric(run.info.run_id, "loss", loss, step=epoch)

        # Log the model with Model Registry
        mlflow.keras.log_model(model, model_name, registered_model_name=model_name)
        
        # save the trained model
        model.save(f"./{model_name}.h5")
        
        # Get model's artifact URI
        #model_artifacts_uri = client.get_artifact_uri(run.info.run_id)
        #print(f"Model's artifact URI: {model_artifacts_uri}")

        
    #client.create_registered_model(model_name)
    desc = "A new version of the model"
    runs_uri = f"runs:/{run.info.run_id}/lstm-model"
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
    mv = client.create_model_version(model_name, model_src, run.info.run_id, description=desc)
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Description: {mv.description}")
    print(f"Status: {mv.status}")
    print(f"Stage: {mv.current_stage}")
    client.set_terminated(run.info.run_id)

    return model

def make_predictions(model, scaler, scaled_data, dataset, training_data_len):
    # Create the testing data set
    test_data = scaled_data[training_data_len - 60 :, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)

    # scale the predict data
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f"rmse: {rmse}")

    return predictions


def plot_results(train_data, valid_data, predictions):
    # Plotting the data
    train = train_data
    valid = valid_data
    valid["Predictions"] = predictions
    plt.figure(figsize=(16, 6))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.plot(train_data["Close"])
    plt.plot(valid_data[["Close", "Predictions"]])
    plt.plot(valid_data["Predictions"])
    plt.legend(["Train", "Val", "Predictions"], loc="lower right")
    plt.show(block=True)


def save_results(
    train_data,
    valid_data,
    predictions,
    experiment_name="Stock_Price_Prediction",
    run_name="Model_Run",
):
    # Plotting the data
    train = train_data
    valid = valid_data
    valid["Predictions"] = predictions
    plt.figure(figsize=(16, 6))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.plot(train_data["Close"])
    plt.plot(valid_data[["Close", "Predictions"]])
    plt.plot(valid_data["Predictions"])
    plt.legend(["Train", "Val", "Predictions"], loc="lower right")

    # Save the plot
    plt.savefig("./result/predictions_plot.png")
    # # plot the data
    # plt.show(block=True)

    # # Log the plot in MLflow
    # with mlflow.start_run(run_name=run_name, experiment_name=experiment_name) as run:
    #     mlflow.log_artifact("predictions_plot.png")

def pipeline():
    stock_code = "AAPL"
    start_date = "2012-01-01"
    end_date = datetime.now()
    flag = 1

    scaler, scaled_data, training_data_len, dataset, data = make_Dataset(
        stock_code, start_date, end_date
    )
    print("make_dataset - success!!")

    x_train, y_train = prepare_training_data(scaled_data, training_data_len)
    print("prepare_training_data - success!!")

    model = build_improved_LSTM_model_with_MLflow(x_train, y_train)
    print("build_improved_LSTM_model_with_MLflow - success!!")

    # predictions = make_predictions(
    #     model, scaler, scaled_data, dataset, training_data_len
    # )
    # print("make_predictions - success!!")

    # save_results(data[:training_data_len], data[training_data_len:], predictions)
    # print("plot_results - success!!")

if __name__ == "__main__":
    pipeline()
