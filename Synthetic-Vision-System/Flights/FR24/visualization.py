import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FlightVisualizer:
    def __init__(self):
        """Инициализация визуализатора"""
        # Создаем окно с графиками
        self.fig = plt.figure(figsize=(15, 10))
        
        # График траектории
        self.ax_trajectory = self.fig.add_subplot(221, projection='3d')
        self.ax_trajectory.set_title('Траектории полета')
        self.ax_trajectory.set_xlabel('Широта')
        self.ax_trajectory.set_ylabel('Долгота')
        self.ax_trajectory.set_zlabel('Высота (м)')
        
        # График скоростей
        self.ax_velocity = self.fig.add_subplot(222)
        self.ax_velocity.set_title('Скорости')
        self.ax_velocity.set_xlabel('Время (с)')
        self.ax_velocity.set_ylabel('Скорость (м/с)')
        self.ax_velocity.grid(True)
        
        # График углов
        self.ax_attitude = self.fig.add_subplot(223)
        self.ax_attitude.set_title('Углы')
        self.ax_attitude.set_xlabel('Время (с)')
        self.ax_attitude.set_ylabel('Угол (град)')
        self.ax_attitude.grid(True)
        
        # График рисков
        self.ax_risks = self.fig.add_subplot(224)
        self.ax_risks.set_title('Риски столкновения')
        self.ax_risks.set_xlabel('Время (с)')
        self.ax_risks.set_ylabel('Вероятность')
        self.ax_risks.grid(True)
        self.ax_risks.set_ylim(0, 1)
        
        # Данные для графиков
        self.velocities = []
        self.attitudes = []
        self.time_points = []
        self.collision_risks = []
        
        # Линии на графиках
        self.actual_line = None
        self.theoretical_line = None
        self.vx_line, = self.ax_velocity.plot([], [], label='Vx')
        self.vy_line, = self.ax_velocity.plot([], [], label='Vy')
        self.vz_line, = self.ax_velocity.plot([], [], label='Vz')
        self.roll_line, = self.ax_attitude.plot([], [], label='Крен')
        self.pitch_line, = self.ax_attitude.plot([], [], label='Тангаж')
        self.heading_line, = self.ax_attitude.plot([], [], label='Курс')
        self.risk_line, = self.ax_risks.plot([], [], 'r-', label='Вероятность')
        
        # Добавляем легенды
        self.ax_velocity.legend()
        self.ax_attitude.legend()
        self.ax_risks.legend()
        
        plt.tight_layout()
        plt.ion()
        plt.show()
        
    def update(self, actual_position, theoretical_positions, actual_positions,
              velocity, roll, pitch, heading, collision_warning=False, risks=None):
        """Обновление визуализации"""
        # Добавляем новые данные
        self.velocities.append(velocity)
        self.attitudes.append([roll, pitch, heading])
        self.time_points.append(len(self.time_points) * 0.1)  # шаг 0.1с
        
        if risks:
            self.collision_risks.append(risks['probability'])
        else:
            self.collision_risks.append(0.0)
        
        # Обновляем график траекторий
        if self.actual_line:
            self.actual_line.remove()
        if self.theoretical_line:
            self.theoretical_line.remove()
            
        # Фактическая траектория
        actual_positions = np.array(actual_positions)
        self.actual_line = self.ax_trajectory.plot(
            actual_positions[:, 0],
            actual_positions[:, 1],
            -actual_positions[:, 2],
            'b-', label='Фактическая',
            linewidth=2
        )[0]
        
        # Теоретическая траектория
        theoretical = np.array(theoretical_positions)
        self.theoretical_line = self.ax_trajectory.plot(
            theoretical[:, 0],
            theoretical[:, 1],
            -theoretical[:, 2],
            'r--', label='Теоретическая',
            linewidth=2,
            alpha=0.7
        )[0]
        
        self.ax_trajectory.legend()
        
        # Обновляем графики скоростей
        velocities = np.array(self.velocities)
        self.vx_line.set_data(self.time_points, velocities[:, 0])
        self.vy_line.set_data(self.time_points, velocities[:, 1])
        self.vz_line.set_data(self.time_points, velocities[:, 2])
        
        # Обновляем графики углов
        attitudes = np.degrees(np.array(self.attitudes))
        self.roll_line.set_data(self.time_points, attitudes[:, 0])
        self.pitch_line.set_data(self.time_points, attitudes[:, 1])
        self.heading_line.set_data(self.time_points, attitudes[:, 2])
        
        # Обновляем график рисков
        self.risk_line.set_data(self.time_points, self.collision_risks)
        
        # Обновляем пределы осей
        for ax in [self.ax_velocity, self.ax_attitude, self.ax_risks]:
            ax.relim()
            ax.autoscale_view()
        
        # Обновляем отображение
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Закрытие визуализатора"""
        plt.close(self.fig) 