import numpy as np
import pandas as pd
from typing import List
from models.mathematical_model import (
    AircraftState,
    IntersectionCalculator,
    TerrainPoint,
    UncertaintyParams
)

class RealFlight:
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: путь к файлу с данными полета
        """
        self.csv_path = csv_path
        self.data = None
        self.current_index = 0
        self.model = IntersectionCalculator(
            UncertaintyParams(
                pos_std=0.5,    # СКО позиции (м)
                vel_std=0.1,    # СКО скорости (м/с)
                att_std=0.01,   # СКО углов (рад)
                terrain_std=1.0  # СКО высоты рельефа (м)
            )
        )
        self.actual_positions = []  # фактические позиции из файла
        self.theoretical_positions = []  # теоретические позиции из модели
        self.collision_warning = False
        self.risks = None
        
        # Загрузка данных
        self.load_data()
        
    def load_data(self):
        """Загрузка данных из CSV файла"""
        self.data = pd.read_csv(self.csv_path, sep=';')
        
    def set_terrain_data(self, terrain_data: List[TerrainPoint]):
        """
        Установка данных о рельефе
        
        Args:
            terrain_data: данные о рельефе в формате списка объектов TerrainPoint
        """
        self.terrain_data = terrain_data
        
    def get_current_state(self) -> dict:
        """Получение текущего состояния полета"""
        if self.current_index >= len(self.data):
            return None
            
        row = self.data.iloc[self.current_index]
        
        # Фактическое состояние из файла
        actual_state = AircraftState(
            x=row['latitude'],
            y=row['longitude'],
            z=-row['altitude'],  # отрицательное значение, т.к. ось Z направлена вниз
            vx=row['Vx'],
            vy=row['Vy'],
            vz=row['Vz'],
            roll=np.radians(row['roll']),
            pitch=np.radians(row['pitch']),
            heading=np.radians(row['heading'])
        )
        
        # Добавляем фактическую позицию
        self.actual_positions.append([
            actual_state.x,
            actual_state.y,
            actual_state.z
        ])
        
        # Если это первая точка, инициализируем теоретическую траекторию
        if self.current_index == 0:
            self.theoretical_positions = []
            theoretical_trajectory = self.model.rk4_trajectory(
                actual_state,
                dt=0.1,
                steps=len(self.data)
            )
            self.theoretical_positions = [pos[0] for pos in theoretical_trajectory]
        
        return {
            'actual_state': actual_state,
            'theoretical_positions': self.theoretical_positions,
            'actual_positions': self.actual_positions,
            'collision_warning': self.collision_warning,
            'risks': self.risks
        }
        
    def predict_trajectory(self, time_horizon: float = 30.0):
        """
        Предсказание траектории и оценка рисков
        
        Args:
            time_horizon: горизонт прогноза (с)
        """
        current_state = self.get_current_state()
        if current_state is None:
            return []
            
        # Проверяем на столкновения для теоретической траектории
        if self.terrain_data is not None:
            # Берем часть теоретической траектории впереди текущей позиции
            future_positions = self.theoretical_positions[self.current_index:]
            future_trajectory = [(pos, np.eye(3) * 0.1) for pos in future_positions]  # добавляем ковариацию
            
            total_prob, _ = self.model.calculate_intersection_probability(
                future_trajectory,
                self.terrain_data
            )
            
            self.collision_warning = total_prob > 0.1  # порог вероятности
            
            if self.collision_warning:
                self.risks = {
                    'probability': total_prob,
                    'time_to_impact': self._estimate_time_to_impact(future_positions)
                }
            else:
                self.risks = None
        
        return self.theoretical_positions
        
    def _estimate_time_to_impact(self, positions: List[np.ndarray]) -> float:
        """Оценка времени до столкновения"""
        if not positions:
            return float('inf')
            
        for i, pos in enumerate(positions):
            # Находим ближайшую точку рельефа
            terrain_points = np.array([[p.lat, p.lon, p.elevation] for p in self.terrain_data])
            distances = np.linalg.norm(terrain_points[:, :2] - pos[:2], axis=1)
            nearest_idx = np.argmin(distances)
            terrain_height = terrain_points[nearest_idx, 2]
            
            # Проверяем высоту
            if -pos[2] <= terrain_height:
                return i * 0.1  # шаг 0.1с
                
        return float('inf')
        
    def step(self):
        """Переход к следующему состоянию"""
        self.current_index += 1
        current_state = self.get_current_state()
        
        if current_state:
            self.predict_trajectory()
            
        return current_state