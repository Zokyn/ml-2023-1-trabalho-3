import pandas as pd
from sklearn.calibration import LabelEncoder
# from sklearn.model_selection import train_test_split

# Recebendo os dados de entrada em csv
data = pd.read_csv('ds_salaries.csv')

# Eliminando as coluans desnecessárias 
X = data.drop(['job_title', 'salary', 'salary_currency'], axis=1)

# Codificando valores categoricos
for object_column in X.select_dtypes('object').columns:
    X[object_column] = LabelEncoder().fit_transform(X[object_column]) 

print(X.info())
print(X.describe())


