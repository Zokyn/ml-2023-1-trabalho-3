import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

class StopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mse')<=0.9):
            print("\nSTOP [Mean Squared Error to low so cancelling training]")
            self.model.stop_training = True

callbacks = StopCallback()
### PRE-PROCESSING ###
# Recebendo os dados de entrada em csv
data = pd.read_csv('ds_salaries.csv')

# Eliminando as coluans desnecessárias 
X = data.drop(['job_title', 'salary', 'salary_currency'], axis=1)
y = data['salary_in_usd']

# Codificando valores categoricos
for object_column in X.select_dtypes('object').columns:
    # Categorizando coluna usando o LabelEncoder 
    enconded_column = LabelEncoder().fit_transform(X[object_column])

    # Transformando a coluna codificada em int64
    X[object_column] = pd.to_numeric(enconded_column).astype('int64')

    # Essa ultima linha podia ser ingnorada mas foi usada afim de 
    # manter todas as colunas em só tipo, fazendo assim com que o 
    # dataset, e posteriormente o modelo lide apenas com um tipo de dados

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
### MODELING ###
def build_model():
    n = X_train.shape[1]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(500, input_dim=n, activation='relu', name='input'))
    model.add(tf.keras.layers.Dense(250, activation='relu', name='hidden_1'))
    model.add(tf.keras.layers.Dense(250, activation='relu', name='hidden_2'))
    model.add(tf.keras.layers.Dense(50, activation='relu', name='hidden_3'))
    model.add(tf.keras.layers.Dense(1, name='output'))

    # model.summary()

    accuracy = tf.keras.metrics.Accuracy()
    model.compile(loss='mse', optimizer='adam', metrics=['mae','mse', accuracy])

    history = model.fit(X_train, y_train, epochs=200, callbacks=[callbacks])

    # Predizendo valores de treinamento
    y_predict = model.predict(X_test)
    # print(X_test[:10]['salary_in_usd'])
    # print(e)

    # plt.scatter(range(len(y_test[:500])), y_test[:500], color='blue', label='real')
    # plt.scatter(range(len(x_predict[:500])), x_predict[:500],color='red', marker='x', label='predict')
    # plt.xlabel('epoch')
    # plt.show()

    # Mostrando historico de perdas e metricas
    h = pd.DataFrame(history.history)
    h['epoch'] = history.epoch
    # print(h.tail())
    return h , X_test, y_predict
def show_predicts(x_predict, y_predict):
        # plt.scatter(range(len(y), np.array(y_test['salary_in_usd'][:10]), color='blue', label='real')
    # plt.scatter(range(len(X_test[:10])), np.array(X_test['salary_in_usd'][:10]) ,color='red', marker='x', label='predict')
    plt.scatter(len(y_predict), y_predict, color='blue', label='real')
    plt.scatter((len(x_predict)), x_predict, color='red', marker='x', label='predict')
    
    plt.show()


def show_results(hist):
    print(f"%"*25)
    print(f"ÉPOCAS TOTAL {hist['epoch'].iloc[-1]}")
    print(f"MAE| {hist['mae'].iloc[-1]}")
    # print(f"MAE| {hist.iloc[hist['mae'][:1].index]}")
    print(f"MSE| {hist['mse'].iloc[-1]}")
    print(f"Accuracy| {hist['accuracy'].iloc[-1]}")
    # print(f"{hist.keys()}")
    print(f"%"*25)

def show_grapichs(hist,):
    plt.figure(figsize=(8, 6))
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Square Error [$MPG^2$]')
    # plt.plot(hist['epoch'], hist['mse'],
    #         label='Train Error')
    # plt.plot(hist['epoch'], hist['val_mse'],
    #         label = 'Val Error')
    # plt.ylim([0,20])
    # plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='MAE')
    plt.plot(hist['epoch'], hist['loss'],
            label = 'Loss Error')
    plt.ylim([0,100])
    plt.legend()

    plt.show()

hist, x, y = build_model()
show_results(hist)
# show_predicts(x, y)
show_grapichs(hist)