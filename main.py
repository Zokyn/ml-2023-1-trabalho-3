import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

### MODELING ###
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
# print(X_train.shape, X_test.shape)

n = X_train.shape[1]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(500, input_dim=n, activation='relu', name='input'))
model.add(tf.keras.layers.Dense(250, activation='relu', name='hidden_1'))
model.add(tf.keras.layers.Dense(50, activation='relu', name='hidden_2'))
model.add(tf.keras.layers.Dense(1, name='output'))

# model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(X_train, y_train, epochs=10)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
#     tf.keras.layers.Dense(156, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', 
#               metrics=[tf.keras.metrics.MeanSquaredError()])

# # # print(model.summary())
# # # print(model.output_shape)

# model.fit(X_train, y_train, batch_size=10, epochs=20)
# # # print(X.info())
# # # print(X.describe())


