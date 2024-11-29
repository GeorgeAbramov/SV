# Импорт необходимых библиотек
import numpy as np  # Для работы с массивами и матрицами
from scipy.stats import multivariate_normal as mvn  # Для многомерного нормального распределения
import matplotlib.pyplot as plt  # Для визуализации данных

# Класс для моделирования 3D траектории с фильтром Калмана
class trajectory3D():
    
    def __init__(self, seed=123, ndat=100, delta_time = 0.1, q=2., r=0.5):
        """
        Конструктор класса
        Параметры:
        seed - зерно для генератора случайных чисел
        ndat - количество точек траектории
        delta_time - шаг по времени между точками
        q - параметр шума процесса
        r - параметр шума измерений
        """        
        self.ndat = ndat  # Сохраняем количество точек
        self.seed = seed  # Сохраняем зерно
        self.q = q  # Сохраняем параметр шума процесса
        self.dt = delta_time  # Сохраняем шаг по времени
        dt = self.dt
        self.r = r  # Сохраняем параметр шума измерений
        
        # Матрица перехода состояния (матрица A)
        # Описывает как система изменяется со временем
        self.A = np.array([
            [1, 0, 0, dt, 0, 0],  # x = x + dx*dt
            [0, 1, 0, 0, dt, 0],  # y = y + dy*dt
            [0, 0, 1, 0, 0, dt],  # z = z + dz*dt
            [0, 0, 0, 1, 0, 0],   # dx = dx
            [0, 0, 0, 0, 1, 0],   # dy = dy
            [0, 0, 0, 0, 0, 1]    # dz = dz
        ])
        
        # Матрица ковариации шума процесса
        self.Q = self.q * np.diag([dt, dt, dt, dt, dt, dt])
        
        # Матрица измерений (матрица H)
        # Связывает состояние системы с измерениями
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Измеряем только координаты
            [0, 1, 0, 0, 0, 0],  # без скоростей
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Матрица ковариации шума измерений
        self.R = self.r**2 * np.eye(3)
        
        # Начальное состояние системы (позиция и скорость)
        self.m0 = np.array([0., 0., 0., 1., 1., 1.])
        
        # Массивы для хранения истинных состояний и измерений
        self.X = np.zeros(shape=(self.A.shape[0], self.ndat))
        self.Y = np.zeros(shape=(self.H.shape[0], self.ndat))
        
        # Запуск симуляции
        self._simulate()
        
    def _simulate(self):
        """
        Метод для симуляции траектории
        Генерирует последовательность состояний и измерений
        """
        np.random.seed(self.seed)  # Устанавливаем зерно для воспроизводимости
        
        x = self.m0  # Начинаем с начального состояния
        for t in range(self.ndat):
            # Генерируем шум процесса
            q = mvn.rvs(cov=self.Q)
            # Вычисляем следующее состояние
            x = self.A.dot(x) + q
            # Генерируем зашумленное измерение
            y = self.H.dot(x) + mvn.rvs(cov=self.R)
            # Сохраняем состояние и измерение
            self.X[:,t] = x.flatten()
            self.Y[:,t] = y.flatten()
            
        # Визуализация результатов
        plt.plot(self.X)  # График истинных состояний
        plt.plot(self.Y)  # График измерений
        plt.show()

# Создание экземпляра класса для тестирования
ter = trajectory3D()