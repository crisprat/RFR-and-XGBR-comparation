import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Считывание данных
X = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Разделение данных обучения и валидации
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Нахождение колонок с не числовыми данными
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Нахождение колонок с числовыми данными
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# pd вместо пайплайнов для упрощения
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


def get_cur_mae(n):
  model = XGBRegressor(n_estimators=n)
  model.fit(X_train, y_train, early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
  preds = model.predict(X_valid)
  return mean_absolute_error(y_valid, preds)

  min_mae = 1000000
  train_n_estim = 0
  for n in range(50, 1001, 50):
    cur_mae = get_cur_mae(n)
    if cur_mae < min_mae:
      min_mae = cur_mae
      train_n_estim = n


# Создание и обучение модели
model = XGBRegressor(n_estimators=train_n_estim)

model.fit(X_train, y_train, early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)

preds = model.predict(X_test)
print("Вероятная величина: ", preds)
print("MAE: ", mean_absolure_error(model.predict(X_valid), y_valid))
