import numpy
# для сигмоиды

import scipy.special
# для реализации матричного умножения
import matplotlib.pyplot
# для отображения mnist в виде изображений
%matplotlib inline
class neuralNetwork:
    
    
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # количество узлов на каждом слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # 
        # матрица весов между входным и скрытым и скрытым и выходным слоями(вначале рандомно)
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # коэффициент обучения
        self.lr = learningrate
        
        # функция активации-сигмоида
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # тренировка сети
    def train(self, inputs_list, targets_list):
        # входные данные переведем в двумерные массивы
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # посчитаем входные данные, поступающие в скрытый слой
        hidden_inputs = numpy.dot(self.wih, inputs)
        # примним функцию активации
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # считаем входные сигналы, поступающие в последний слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # ф-я активации
        final_outputs = self.activation_function(final_inputs)
        
        # ошибка выходного слоя-это разницы между ошибкой входного слоя и тем значением, которое должно быть
        output_errors = targets - final_outputs
        # ошибки для скрытого слоя-это ошибки выходного слоя, распределенные согласно весам дуг из скрытого слоя в выходной
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # веса между скрытым и выходным слоем пересчитываются с учетом ошибки
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # веса между входным и скрытым слоем тоже пересчитываются
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # опрос сети
    def query(self, inputs_list):
        # перевод входных данных в двумерный формат
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # считаем данные, поступающие в скрытый слой
        hidden_inputs = numpy.dot(self.wih, inputs)
        # считаем данные, поступающие из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # считаем данные, поступающие в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # считаем данные, поступающие из выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1


n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
# загрузка тренировочкого датасета (10000 образцов)
training_data_file = open("mnist_dataset/mnist_train_1.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# кол-вл эпох
epochs = 5

for e in range(epochs):
    #
    for record in training_data_list:
        # csv формат, поэтому убираем ,
        all_values = record.split(',')
        # скалирование цвета пикселя
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # инициализируем то, с чем будем сравнивать выходные данные
        targets = numpy.zeros(output_nodes) + 0.01
        # у нас 10 выходов
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
# загружаем датасет для тестирования (10)
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# для отображения, сколько изображений мы распознали правильно
scorecard = []

# g
for record in test_data_list:
    
    all_values = record.split(',')
    # правильный ответ-это первое значение в csv-файле
    correct_label = int(all_values[0])
    # скалирование необходимо, чтобы цвет каждого пикселя был в диапазоне 0..0.99
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
   
    outputs = n.query(inputs)
    # находим выход с самым большим значением, он соответствует паттерну
    label = numpy.argmax(outputs)
    # tckb ghfdbkmyj hfcgjpyfkb
    if (label == correct_label):
        # 
        scorecard.append(1)
    else:
        # если неправильно распознали
        scorecard.append(0)
        pass
    
    pass
# процент корректно распознанных картинок по отношению ко всем изображениям
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
data_file = open("mnist_dataset/mnist_train.csv", 'r') 
data_list = data_file.readlines() 
data_file.close()
data_list[0]
all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28)) 
matplotlib.pyplot.imshow(image_array, cmap='Greys',
interpolation='None')
print(n.query((numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01))
print(scorecard)
