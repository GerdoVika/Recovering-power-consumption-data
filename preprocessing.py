import pandas as pd
import numpy as np


def make_harmonic_features(val, period):
  '''Получить sin и cos переменной val с периодом period'''
  val = val * 2 * np.pi / period
  return np.sin(val), np.cos(val)

'''
  Предобработка и генерация новых признаков

  аргументы:
  df -- DataFrame, столбцы ['Дата', 'A+']
  возвращает:
  X -- np.array массив признаков
  Y -- np.array массив целевых значений
  '''
def get_time_feature(df):
  '''Выделение пропусков

    аргументы:
    df -- DataFrame, столбцы ['Дата', 'A+']
    возвращает:
    train_data -- DataFrame, столбцы ['Дата', 'A_prev', 'A+']
    target_data -- DataFrame, столбцы ['Дата', 'A_prev', 'A+']
    '''
  month = df['Дата'].dt.month.to_numpy()
  dayofweek = df['Дата'].dt.dayofweek.to_numpy()
  time = df['Дата'].dt.hour.to_numpy() + df['Дата'].dt.minute.to_numpy() / 60

  sin_month, cos_month = make_harmonic_features(month, 30)
  sin_day, cos_day = make_harmonic_features(dayofweek, 7)
  sin_time, cos_time = make_harmonic_features(time, 24)

  df = pd.DataFrame({'sin_month': sin_month, 'cos_month': cos_month,
                     'sin_day': sin_day, 'cos_day': cos_day, 'sin_time': sin_time,
                     'cos_time': cos_time, 'A+': df['A+'], 'A_prev': df['A_prev']})

  Y = df['A+'].to_numpy()
  X = df.drop('A+', axis=1).to_numpy()
  return X, Y


def preprocess_data(df):
  df.loc[(df['A+'] == '-'), 'A+'] = np.NaN
  A_mean = df['A+'].mean()
  A_prev = df['A+'][:-1]
  all_data = df.reset_index(drop=True);
  df = df[1:].reset_index(drop=True)
  df.insert(loc=2, column='A_prev', value=A_prev)
  all_data.insert(loc=2, column='A_prev', value=A_prev)

  train_data = df[df['A+'].notna()]
  train_data = train_data[train_data['A_prev'].notna()]
  target_data = df[df['A+'].isna()]
  target_data.index = target_data.index+1
  all_data['A_prev'].fillna(A_mean, inplace=True)
  all_data = all_data[all_data['A+'].notna()]

  return all_data,train_data, target_data


def read_file(filename):
  '''Чтение данных из файла'''
  df = pd.read_excel(filename, header=8, usecols='A,D', parse_dates=True)
  return df