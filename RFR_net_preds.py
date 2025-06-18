import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Считывание данных
train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')

# Очистка данных
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Разделение данных для обучения и проверки
X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# OneHot - подстановка данных
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index


num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

# Подбор глубины обучения
def get_cur_mae(n):
  model = RandomForestRegressor(n_estimators=n, random_state=0)
  model.fit(OH_X_train, y_train)
  preds = model.predict(OH_X_valid)
  return mean_absolute_error(y_valid, preds)


min_mae = 1000000
train_n_estim = 0
for n in range(100, 1001, 50):
  cur_mae = get_cur_mae(n)
  if cur_mae < min mae:
    train_n_estim = n
    min_mae = cur_mae

# Cоздание и обучение модели
model = RandomForestRegressor(n_estimators=train_n_estim, random_state=0)
model.fit(OH_X_train, y_train)
print("Вероятная величина: ", model.predict(test_data))
print("MAE: ", mean_absolure_error(model.predict(OH_X_valid), y_valid))
