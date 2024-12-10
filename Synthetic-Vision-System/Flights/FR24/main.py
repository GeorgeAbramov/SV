import time
import numpy as np
import pandas as pd
from pathlib import Path
from real_flight import RealFlight
from visualization import FlightVisualizer
from models.mathematical_model import TerrainPoint

def load_terrain_data():
    """
    Загрузка данных о рельефе на основе траектории полета
    """
    # Читаем данные о траектории
    csv_path = Path(__file__).parent / 'navdata-for-parsing.csv'
    flight_data = pd.read_csv(csv_path, sep=';')
    
    # Определяем границы области
    lat_min = flight_data['latitude'].min() - 0.001  # примерно 100м запас
    lat_max = flight_data['latitude'].max() + 0.001
    lon_min = flight_data['longitude'].min() - 0.001
    lon_max = flight_data['longitude'].max() + 0.001
    
    # Создаем сетку точек рельефа
    lat_points = np.linspace(lat_min, lat_max, 20)
    lon_points = np.linspace(lon_min, lon_max, 20)
    
    terrain_points = []
    for lat in lat_points:
        for lon in lon_points:
            # Находим ближайшую точку траектории
            distances = np.sqrt(
                (flight_data['latitude'] - lat)**2 + 
                (flight_data['longitude'] - lon)**2
            )
            nearest_idx = distances.argmin()
            
            # Базовая высота из траектории
            base_elevation = flight_data['altitude'].iloc[nearest_idx] - 50  # 50м запас до земли
            
            # Добавляем рельеф (пример генерации холмистой местности)
            elevation = max(0, base_elevation + 
                          10 * np.sin(lat * 100) * np.cos(lon * 100))  # синусоидальные холмы
            
            terrain_points.append(
                TerrainPoint(lat=lat, lon=lon, elevation=elevation)
            )
    
    return terrain_points

def main():
    # Путь к файлу с данными
    data_path = Path(__file__).parent / 'navdata-for-parsing.csv'
    
    # Инициализация компонентов
    flight = RealFlight(data_path)
    visualizer = FlightVisualizer()
    
    # Загружаем данные о рельефе
    terrain_data = load_terrain_data()
    flight.set_terrain_data(terrain_data)
    
    # Основной цикл симуляции
    try:
        while True:
            # Получаем текущее состояние
            current_state = flight.get_current_state()
            if current_state is None:
                print("Симуляция завершена")
                break
            
            # Обновляем визуализацию
            state = current_state['actual_state']
            visualizer.update(
                actual_position=[state.x, state.y, -state.z],
                theoretical_positions=current_state['theoretical_positions'],
                actual_positions=current_state['actual_positions'],
                velocity=[state.vx, state.vy, state.vz],
                roll=state.roll,
                pitch=state.pitch,
                heading=state.heading,
                collision_warning=current_state['collision_warning'],
                risks=current_state['risks']
            )
            
            # Выводим предупреждения
            if current_state['collision_warning']:
                print("\033[91mВНИМАНИЕ! Опасность столкновения!")
                if current_state['risks']:
                    print(f"Вероятность столкновения: {current_state['risks']['probability']:.3f}")
                    print(f"Время до столкновения: {current_state['risks']['time_to_impact']:.1f}с")
                print("\033[0m")
            
            # Переходим к следующему состоянию
            flight.step()
            
            # Задержка для реального времени
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nЗавершение симуляции...")
    finally:
        visualizer.close()

if __name__ == "__main__":
    main() 