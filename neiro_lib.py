# -*- coding: utf-8 -*-
"""
Функции для работы с нейросетью
ver 1.2
"""
import torch
import torch.nn as nn
import numpy as np


count_parametrs  = 7 #(количество параметров)

# структура нейросети
def init_model() :
  model = torch.nn.Sequential(
  torch.nn.Linear(count_parametrs, 100),
  torch.nn.ReLU(),
  torch.nn.Linear(100, 100),
  torch.nn.ReLU(),
  torch.nn.Linear(100, 100),
  torch.nn.ReLU(),
  torch.nn.Linear(100, 1))
  return model


"""
загрузка весов нейросети
аргументы:
model - модель
path - путь к модели. Должен содержать в конце \model
"""
def load_model(path, model):
  model.load_state_dict(torch.load(path))
  model.eval()
  return model

# сохранение весов нейросети
def save_model(path, model):
  torch.save(model.state_dict(), path)


"""
Предсказание для экземпляра
аргументы:
model - модель
model - модель
X - вектор признаков
"""
def predict_model(X, model):
    norm_x = X;
    predict_data = torch.from_numpy(norm_x).float()
    predict_y = float(model(predict_data))
    Y =  predict_y
    return Y

"""
Предсказание для экземпляра
аргументы:
model - модель
model - модель
X_train,Y_train - выборка для обучения
"""
def train_model(X_train,Y_train,model):
  err = 10;

  # Вводим функцию ошибки
  criterion = torch.nn.MSELoss(reduction='mean')
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  #Перевод из типа nympy array в tensor pycharm
  X_train_torch = torch.from_numpy(X_train).float()
  Y_train_torch = torch.from_numpy(Y_train).float()
  # Вычисление через видеокарту
  if torch.cuda.is_available():
    model = model.cuda()
    X_train_torch, Y_train_torch = X_train_torch.cuda(), Y_train_torch.cuda()
  output = model(X_train_torch[1,:])
  loss = criterion(output, torch.reshape(Y_train_torch, (len(Y_train_torch), 1)))
  epoch = 0 #Количество итераций
  print("Производится обучение")
  while loss>=err:
    epoch = epoch+1
    output = model(X_train_torch)
    loss = criterion(output, torch.reshape(Y_train_torch, (len(Y_train_torch), 1)))
    # if (np.mod(epoch, 10) == 0):
    # print('Epoch: ', epoch, 'Loss: ', loss.item()) #Вывод в консоль точность впроцессе обучения
    optimizer.zero_grad()#Шаг по градиентному спуску
    loss.backward()
    optimizer.step()
    #Ограничение на количество итераций
    if epoch>=1000:
      break
  # Сохранение модели после обучения
  save_model(__file__[0:-13] + "\model", model)


#при желании создать пустую нейросеть - выполниить файл neiro_lib.py
if __name__ == '__main__':
  model = init_model()
  save_model(__file__[0:-13]+"\model",model)

