#import neptune
#from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import keras_tuner as kt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn import metrics
import matplotlib.pyplot as plt 
#import neptunecontrib.monitoring.kerastuner as npt_utils
from tensorflow.keras.optimizers import Adam

#neptune.init(project_qualified_name='kehoffman3/nba-model',
#                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmM2IwMjY5NC0yOWE4LTQ0NzgtYjY5NS1mNTQyMDllZjc1MDEifQ==') # your credentials

#neptune.create_experiment('nba-bayesian-sweep')

def get_train_test_data(cutoff):
    data = pd.read_excel('../Datasets/Full-Data-Set-UnderOver-ML-Odds-2020-21.xlsx')
    data['Season'] = data["Date"].str.split('-').str[2].astype(int)

    
    train_df = data[data["Season"] < cutoff ]
    test_df = data[data["Season"] >= cutoff ]

    #scores = train_df['Score']
    margin = train_df['Home-Team-Win']

    columns_to_drop = ['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1', 'OU', 'OU-Cover', 'Home-Odds', 'Away-Odds', 'Season']

    train_df.drop(columns_to_drop, axis=1, inplace=True)

    train_df = train_df.values.astype(float)

    x_train = tf.keras.utils.normalize(train_df, axis=1)
    y_train = np.asarray(margin)

    margin = test_df['Home-Team-Win']

    test_df.drop(columns_to_drop, axis=1, inplace=True)

    test_df = test_df.values.astype(float)

    x_test = tf.keras.utils.normalize(test_df, axis=1)
    y_test = np.asarray(margin)
    return data, x_train, x_test, y_train, y_test


def build_tuner_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    for i in range(hp.Int("num_layers", 2, 10)):
        model.add(
            layers.Dense(
                units=hp.Int("units_" + str(i), min_value=8, max_value=256, step=8),
                activation="relu",
            )
        )
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def tune_model(x_train, y_train, x_test):
    tuner = kt.BayesianOptimization(hypermodel = build_tuner_model,
                     objective = 'val_accuracy',
                     max_trials=200,
					 project_name='hyperband_tuner_2',
                     logger = npt_utils.NeptuneLogger())
    earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='min')
    batch_size = 32
    tuner.search(x_train,y_train,
             epochs=100,
             validation_split=0.1,
             batch_size=32,
             shuffle=True,
             verbose=1,
             initial_epoch=0,
             callbacks=[earlyStopping],
             use_multiprocessing=True,
             workers=8)
    npt_utils.log_tuner_info(tuner)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
    model.fit(x_train, y_train, epochs=500, validation_split=0.1, batch_size=batch_size,
            callbacks=[earlyStopping])
    return model.predict(x_test)


def build_model_and_predict(x_train, y_train, x_test):
    current_time = str(time.time())

    #tensorboard = TensorBoard(log_dir='../Logs/{}'.format(current_time))
    earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('../Models/Final-Model-ML-' + current_time, save_best_only=True, monitor='val_loss', mode='min')
    #tensorboard = TensorBoard(log_dir='../Logs/{}'.format(current_time))
    #neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    # model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu6))

    activation = tf.keras.activations.selu
    model.add(tf.keras.layers.Dense(64, activation=activation))
    model.add(tf.keras.layers.Dense(32, activation=activation))
    model.add(tf.keras.layers.Dense(16, activation=activation))
    model.add(tf.keras.layers.Dense(8, activation=activation))
    model.add(tf.keras.layers.Dense(4, activation=activation))
    # model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu6))

    
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    batch_size = 32
    optimizer = 'adam'
    # run['parameters'] = {
    #     'batch_size': batch_size,
    #     'activation': activation.__name__,
    #     'optimizer': optimizer
    # }

    # run['layers'] = "64|32|16|8|4"

    model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=500, validation_split=0.1, batch_size=batch_size,
            callbacks=[mcp_save, earlyStopping])

    return model.predict(x_test)


def get_bet_results(Y_pred, Y_test, test_df):
    profits = []
    dates = []
    for idx,y in enumerate(Y_test):
        odds = int(test_df.iloc[idx]['Home-Odds']) if Y_pred[idx] else int(test_df.iloc[idx]['Away-Odds'])
        d = test_df.iloc[idx]['Date']
        #print(d)
        date = test_df.iloc[idx]['Date'][:-3] if int(test_df.iloc[idx]['Date'].split('-')[0]) > 9 else d[:-7] + d[-2:]
        #print(date)
        profit = 0
        # Try skipping long odds
        if odds < -300:
             continue
        if y - Y_pred[idx] == 0:
            profit = odds/100 if odds > 0 else -100/odds
        else:
            profit = -1
        profits.append(profit)
        #print(date)
        dates.append(date)
    profit_cum = np.cumsum(profits)  # returns a numpy.ndarray
    data = {
        "dates": dates,
        "profit": profits
    }
    agg = pd.DataFrame(data).groupby(['dates']).agg({"profit":"sum","dates":"count"}).rename({"dates":"dates_count"}, axis=1)
    agg.reset_index(level=0, inplace=True)
    agg["dates"] = pd.to_datetime(agg['dates'])
    agg.sort_values(by=["dates"], inplace=True)
    agg["cumulative"] = np.cumsum(agg["profit"])
    
    # plt.figure()
    # _ = agg.plot(x="dates", y="cumulative")
    # plt.ylabel('units')
    # plt.show()
    accuracy = metrics.accuracy_score(Y_pred, Y_test)
    total_profit = agg['cumulative'].iloc[-1]
    best_7_days =  agg['profit'].rolling(7).sum().max()
    worst_7_days =  agg['profit'].rolling(7).sum().min()
    best_30_days = agg['profit'].rolling(30).sum().max()
    worst_30_days = agg['profit'].rolling(30).sum().min()

    print(accuracy)
    print(total_profit)
    print(best_7_days)
    print(worst_7_days)
    print(best_30_days)
    print(worst_30_days)

    # neptune.log_metric('accuracy', accuracy)
    # neptune.log_metric('total_profit', total_profit)
    # neptune.log_metric('best_7_days', best_7_days)
    # neptune.log_metric('worst_7_days', worst_7_days)
    # neptune.log_metric('best_30_days', best_30_days)
    # neptune.log_metric('worst_30_days', worst_30_days)

    # run["metrics/accuracy"] = accuracy
    # run["metrics/total_profit"] = total_profit
    # run["metrics/best_7_days"] =  best_7_days
    # run["metrics/worst_7_days"] =  worst_7_days
    # run["metrics/best_30_days"] = best_30_days
    # run["metrics/worst_30_days"] = worst_30_days
    return agg

if __name__ == '__main__':
    cutoff = 2018
    data, x_train, x_test, y_train, y_test = get_train_test_data(cutoff)
    #y_pred = tune_model(x_train, y_train, x_test)
    y_pred = build_model_and_predict(x_train, y_train, x_test)
    y_pred = np.argmax(y_pred, axis=1)
    _ = get_bet_results(y_pred, y_test, data[data["Season"] >= cutoff ])
