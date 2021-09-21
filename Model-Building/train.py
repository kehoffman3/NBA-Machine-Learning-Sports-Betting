import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn import metrics
import matplotlib.pyplot as plt 

run = neptune.init(project='kehoffman3/nba-model',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmM2IwMjY5NC0yOWE4LTQ0NzgtYjY5NS1mNTQyMDllZjc1MDEifQ==') # your credentials

def get_train_test_data(cutoff):
    data = pd.read_excel('../Datasets/Full-Data-Set-UnderOver-ML-Odds-2020-21.xlsx')
    data['Season'] = data["Date"].str.split('-').str[2].astype(int)

    
    train_df = data[data["Season"] < cutoff ]
    test_df = data[data["Season"] >= cutoff ]

    #scores = train_df['Score']
    margin = train_df['Home-Team-Win']

    train_df.drop(['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1', 'OU', 'OU-Cover', 'Home-Odds', 'Away-Odds'], axis=1, inplace=True)

    train_df = train_df.values.astype(float)

    x_train = tf.keras.utils.normalize(train_df, axis=1)
    y_train = np.asarray(margin)

    margin = test_df['Home-Team-Win']

    test_df.drop(['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1', 'OU', 'OU-Cover', 'Home-Odds', 'Away-Odds'], axis=1, inplace=True)

    test_df = test_df.values.astype(float)

    x_test = tf.keras.utils.normalize(test_df, axis=1)
    y_test = np.asarray(margin)
    return data, x_train, x_test, y_train, y_test


def build_model_and_predict(x_train, y_train, x_test):
    current_time = str(time.time())

    #tensorboard = TensorBoard(log_dir='../Logs/{}'.format(current_time))
    earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
    #mcp_save = ModelCheckpoint('../Models/Trained-Model-ML-' + current_time, save_best_only=True, monitor='val_loss', mode='min')
    tensorboard = TensorBoard(log_dir='../Logs/{}'.format(current_time))
    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    # model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu6))

    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu6))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu6))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu6))
    model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu6))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu6))
    # model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu6))

    
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    batch_size = 32
    run['parameters'] = {
        'batch_size': batch_size
    }

    run['layers'] = "64|D.5|32|D.5|16|8|4"

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=500, validation_split=0.1, batch_size=batch_size,
            callbacks=[neptune_cbk, earlyStopping, tensorboard])

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
    run["metrics/accuracy"] = metrics.accuracy_score(Y_pred, Y_test)
    run["metrics/total_profit"] = agg['cumulative'].iloc[-1]
    run["metrics/best_7_days"] =  agg['profit'].rolling(7).sum().max()
    run["metrics/worst_7_days"] =  agg['profit'].rolling(7).sum().min()
    run["metrics/best_30_days"] = agg['profit'].rolling(30).sum().max()
    run["metrics/worst_30_days"] = agg['profit'].rolling(30).sum().min()
    return agg

if __name__ == '__main__':
    cutoff = 2016
    data, x_train, x_test, y_train, y_test = get_train_test_data(cutoff)
    y_pred = build_model_and_predict(x_train, y_train, x_test)
    y_pred = np.argmax(y_pred, axis=1)
    _ = get_bet_results(y_pred, y_test, data[data["Season"] >= cutoff ])
