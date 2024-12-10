import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.stats import norm

@dataclass
class TerrainPoint:
    """Точка рельефа"""
    lat: float      # широта (градусы)
    lon: float      # долгота (градусы)
    elevation: float # высота (м)

@dataclass
class AircraftState:
    """Состояние воздушного судна"""
    x: float      # координата X (широта)
    y: float      # координата Y (долгота)
    z: float      # высота (м)
    vx: float     # скорость по X (м/с)
    vy: float     # скорость по Y (м/с)
    vz: float     # вертикальная скорость (м/с)
    roll: float   # крен (рад)
    pitch: float  # тангаж (рад)
    heading: float # курс (рад)

@dataclass
class UncertaintyParams:
    """Параметры неопределенности измерений"""
    pos_std: float = 0.5    # СКО позиции (м)
    vel_std: float = 0.1    # СКО скорости (м/с)
    att_std: float = 0.01   # СКО углов (рад)
    terrain_std: float = 1.0 # СКО высоты рельефа (м)

class IntersectionCalculator:
    """Калькулятор пересечения траектории с рельефом"""
    
    def __init__(self, uncertainty: UncertaintyParams):
        self.uncertainty = uncertainty
        self.g = 9.81  # ускорение свободного падения
        
    def rk4_trajectory(self, 
                      state: AircraftState,
                      dt: float,
                      steps: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Расчет траектории методом Рунге-Кутта 4-го порядка
        
        Args:
            state: начальное состояние
            dt: шаг по времени (с)
            steps: количество шагов
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: список кортежей (среднее, ковариация)
        """
        # Начальное состояние
        x = np.array([state.x, state.y, state.z])
        v = np.array([state.vx, state.vy, state.vz])
        
        # Матрица поворота из связанной в земную СК
        R = self._rotation_matrix(state.roll, state.pitch, state.heading)
        
        # Начальная ковариационная матрица
        P = np.diag([
            self.uncertainty.pos_std**2,
            self.uncertainty.pos_std**2,
            self.uncertainty.pos_std**2,
            self.uncertainty.vel_std**2,
            self.uncertainty.vel_std**2,
            self.uncertainty.vel_std**2
        ])
        
        # Матрица процесса
        F = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        
        # Матрица шума процесса
        Q = np.diag([
            dt**4/4 * self.uncertainty.pos_std**2,
            dt**4/4 * self.uncertainty.pos_std**2,
            dt**4/4 * self.uncertainty.pos_std**2,
            dt**2 * self.uncertainty.vel_std**2,
            dt**2 * self.uncertainty.vel_std**2,
            dt**2 * self.uncertainty.vel_std**2
        ])
        
        # Прогноз распределений
        distributions = []
        
        for _ in range(steps):
            # RK4 для позиции и скорости
            k1_x = v
            k1_v = self._acceleration(x, v, R)
            
            k2_x = v + 0.5*dt*k1_v
            k2_v = self._acceleration(x + 0.5*dt*k1_x, v + 0.5*dt*k1_v, R)
            
            k3_x = v + 0.5*dt*k2_v
            k3_v = self._acceleration(x + 0.5*dt*k2_x, v + 0.5*dt*k2_v, R)
            
            k4_x = v + dt*k3_v
            k4_v = self._acceleration(x + dt*k3_x, v + dt*k3_v, R)
            
            # Обновление состояния
            x = x + (dt/6)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
            v = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
            
            # Обновление ковариации
            P = F @ P @ F.T + Q
            
            # Сохраняем распределение
            distributions.append((x, P[:3, :3]))
            
        return distributions
        
    def _acceleration(self, x: np.ndarray, v: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Расчет ускорения"""
        # Гравитационное ускорение в земной СК
        g_earth = np.array([0, 0, self.g])
        
        # Преобразование в связанную СК
        g_body = R @ g_earth
        
        # Добавляем случайные возмущения
        noise = np.random.normal(0, self.uncertainty.vel_std, 3)
        
        return g_body + noise
        
    def calculate_intersection_probability(self,
                                        trajectory_dist: List[Tuple[np.ndarray, np.ndarray]],
                                        terrain_points: List[TerrainPoint]) -> Tuple[float, List[float]]:
        """
        Расчет вероятности пересечения траектории с рельефом
        
        Args:
            trajectory_dist: распределения положения ВС
            terrain_points: точки рельефа
            
        Returns:
            Tuple[float, List[float]]: (общая вероятность, вероятности по точкам)
        """
        # Преобразуем точки рельефа в массив
        terrain_array = np.array([[p.lat, p.lon, p.elevation] for p in terrain_points])
        
        # Для каждой точки траектории
        intersection_probs = []
        
        for pos_mean, pos_cov in trajectory_dist:
            # Находим ближайшие точки рельефа
            distances = np.linalg.norm(terrain_array[:, :2] - pos_mean[:2], axis=1)
            nearest_idx = np.argsort(distances)[:4]  # берем 4 ближайшие точки
            
            # Интерполируем высоту рельефа и его неопределенность
            weights = 1 / (distances[nearest_idx] + 1e-6)
            weights /= np.sum(weights)
            
            terrain_height = np.sum(weights * terrain_array[nearest_idx, 2])
            terrain_var = self.uncertainty.terrain_std**2
            
            # Разность высот (с учетом того, что ось Z направлена вниз)
            height_diff = -pos_mean[2] - terrain_height
            height_var = pos_cov[2, 2] + terrain_var
            
            # Вероятность пересечения (высота ВС ниже рельефа)
            prob = norm.cdf(0, loc=height_diff, scale=np.sqrt(height_var))
            intersection_probs.append(prob)
            
        # Общая вероятность (хотя бы одно пересечение)
        total_prob = 1 - np.prod(1 - np.array(intersection_probs))
        
        return total_prob, intersection_probs
        
    @staticmethod
    def _rotation_matrix(roll: float, pitch: float, heading: float) -> np.ndarray:
        """Матрица поворота из связанной в земную СК"""
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sh, ch = np.sin(heading), np.cos(heading)
        
        return np.array([
            [ch*cp, sh*cp, -sp],
            [ch*sp*sr - sh*cr, sh*sp*sr + ch*cr, cp*sr],
            [ch*sp*cr + sh*sr, sh*sp*cr - ch*sr, cp*cr]
        ])