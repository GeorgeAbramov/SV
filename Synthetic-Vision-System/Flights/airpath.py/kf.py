import numpy as np

class KF():
    def __init__(self, A, B, H, R, Q):
        """
        Инициализация фильтра Калмана
        A - матрица перехода состояния
        B - матрица управления 
        H - матрица измерений
        R - матрица ковариации шума измерений
        Q - матрица ковариации шума процесса
        """
        self.A = A
        self.B = B
        self.H = np.atleast_2d(H)  # Преобразуем H в 2D массив
        self.Q = Q
        self.P = np.eye(A.shape[0]) * 1000.  # Начальная ковариация ошибки
        self.x = np.zeros(A.shape[0])  # Начальное состояние
        self.log_x = []  # Список для хранения истории состояний
        self.xi = np.zeros(np.asarray(self.P.shape) + 1)  # Информационная матрица
        if np.isscalar(R):
            self.Rinv = 1/R  # Обратная величина R для скаляра
        else:
            self.Rinv = np.linalg.inv(R)  # Обратная матрица R

        
    def predict(self, u=None):
        """
        Шаг предсказания
        u - вектор управления (опционально)
        """
        xminus = self.A.dot(self.x)  # Предсказание состояния
        if u is not None:
            xminus += self.B.dot(u)  # Добавляем влияние управления
        Pminus = self.A.dot(self.P).dot(self.A.T) + self.Q  # Предсказание ковариации
        xi_vector = np.r_[xminus[np.newaxis], np.eye(self.x.shape[0])]
        self.xi = xi_vector.dot(np.linalg.inv(Pminus)).dot(xi_vector.T)
        self.x = xminus   # Временное обновление состояния
        self.P = Pminus   # Временное обновление ковариации
        
    def update(self, y, Rinv=None):
        """
        Шаг коррекции
        y - измерение
        Rinv - обратная матрица ковариации шума измерений (опционально)
        """
        if Rinv is None:
            Rinv = self.Rinv
        y = np.atleast_2d(y).reshape(self.H.shape[0], -1)  # Преобразуем измерение в нужную форму
        T_vector = np.concatenate((y, self.H), axis=1)  
        T = T_vector.T.dot(Rinv).dot(T_vector)
        self.xi += T  # Обновляем информационную матрицу
        self.P = np.linalg.inv(self.xi[1:,1:])  # Обновляем ковариацию
        self.x = self.P.dot(self.xi[1:,0])  # Обновляем состояние
        
    def log(self):
        """Сохраняем текущее состояние в историю"""
        self.log_x.append(self.x.copy())


#--------------
if __name__ == '__main__':
    from scipy.stats import multivariate_normal as mvn
    from scipy.stats import norm
    import matplotlib.pylab as plt
    
    test = 'bivariate'
    
    if test == 'univariate':
        ndat = 100                               # Количество точек данных
        h0 = 0                                   # Начальная высота [м]
        v0 = 520                                 # Начальная скорость [м/с]
        g = 9.81                                 # Ускорение свободного падения [м/с^2]
        dt = 1                                   # Временной шаг [с]
        A = np.array([[1, dt], [0, 1]])         # Матрица перехода состояния
        B = np.array([-.5*dt, -dt])             # Матрица управления
        u = g                                    # Управляющее воздействие (гравитация)
        var_wht = 10                            # Дисперсия шума высоты
        var_wvt = 10                            # Дисперсия шума скорости
        var_y = 90000                           # Дисперсия шума измерений
        
        x = np.zeros((2, ndat))
        x[:,0] = [h0, v0]
        y = np.zeros(ndat)
        
        for t in range(1, ndat):
            x[:,t] = A.dot(x[:,t-1]) + B.dot(u)
            x[:,t] += mvn.rvs(cov=np.diag([var_wht, var_wvt]))
            y[t] = x[0,t] + norm.rvs(scale=np.sqrt(var_y))
    
    #---------
        Q = np.diag([var_wht, var_wvt])         # Матрица ковариации шума процесса
        R = var_y                               # Ковариация шума измерений
        H = np.array([1, 0])                    # Матрица измерений
        
        filtr = KF(A=A, B=B, H=H, R=R, Q=Q)    # Создаем фильтр Калмана
        for yt in y:
            filtr.predict(u)
            filtr.update(yt)
            filtr.log()
        log_x = np.array(filtr.log_x).T
    #----------        
        plt.figure(figsize=(14,4))
        plt.subplot(221)
        plt.plot(x[0], label='Истинная высота')
        plt.plot(y, '+r', label='Измеренная высота')
        plt.legend()
        plt.xlabel('Время')
        plt.subplot(222)
        plt.plot(x[1], label='Истинная скорость')
        plt.plot(np.abs(x[1]), '--b', label='Модуль скорости')
        plt.legend()
        plt.xlabel('Время')
        plt.subplot(223)
        plt.plot(x[0])
        plt.plot(log_x[0])
        plt.subplot(224)
        plt.plot(x[1])
        plt.plot(log_x[1])
        plt.show()
    #==================================
    
    elif test == 'bivariate':
        q = 5.                                  # Интенсивность шума процесса
        dt = .1                                 # Шаг по времени
        r = .2                                  # Интенсивность шума измерений
        A = np.array([[1, 0, dt, 0],           # Матрица перехода состояния
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])
        Q = q * np.array([[dt**3/3, 0      , dt**2/2, 0      ],  # Матрица ковариации шума процесса
                          [0,       dt**3/3, 0,       dt**2/2],
                          [dt**2/2, 0,       dt,      0      ],
                          [0,       dt**2/2, 0,       dt     ]])
        H = np.array([[1., 0, 0, 0],          # Матрица измерений
                      [0., 1, 0, 0]])
        R = r**2 * np.eye(2)                   # Матрица ковариации шума измерений
        m0 = np.array([0., 0., 1., -1.]).reshape(4,1)  # Начальное состояние
        P0 = np.eye(4)                         # Начальная ковариация
        
        # Симуляция траектории
        np.random.seed(1234567890)
        steps = 100                            # Количество шагов
        X = np.zeros(shape=(A.shape[0], steps))  # Истинные состояния
        Y = np.zeros(shape=(H.shape[0], steps))  # Зашумленные измерения
        Y_clear = np.zeros(shape=(H.shape[0], steps))  # Чистые измерения
        x = m0
        for t in range(steps):
            q = np.linalg.cholesky(Q).dot(np.random.normal(size=(A.shape[0], 1)))
            x = A.dot(x) + q
            y_clear = np.dot(H, x)
            y = y_clear + r * np.random.normal(size=(2,1))
            X[:,t] = x.flatten()
            Y[:,t] = y.flatten()
            Y_clear[:,t] = y_clear.flatten()
        
        # Применение фильтра Калмана
        kf = KF(A=A, B=None, H=H, R=R, Q=Q)
        for y in Y.T:
            kf.predict()
            kf.update(y)
            kf.log()
            
            log_x = np.array(kf.log_x).T
            
        # Визуализация результатов
        plt.figure(figsize=(10,10))
        plt.plot(X[0], X[1], '-', label='Траектория')
        plt.plot(Y[0], Y[1], '.', label='Измерения')
        plt.plot(log_x[0], log_x[1], '-', color='red', label='Оценка')
        plt.grid(True)
        plt.legend()