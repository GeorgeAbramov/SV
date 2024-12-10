import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.mathematical_model import (
    AircraftState,
    TerrainPoint,
    UncertaintyParams,
    IntersectionCalculator
)

def analyze_intersection_probability():
    """Анализ пересечения траекторий с рельефом"""
    
    # Создаем тестовый рельеф
    terrain_points = [
        TerrainPoint(lat=55.5624, lon=37.9777, elevation=0),
        TerrainPoint(lat=55.5625, lon=37.9777, elevation=10),
        TerrainPoint(lat=55.5624, lon=37.9778, elevation=5),
        TerrainPoint(lat=55.5625, lon=37.9778, elevation=15)
    ]
    
    # Теоретическая траектория (без погрешностей)
    theoretical_state = AircraftState(
        x=55.5624,  # широта
        y=37.9777,  # долгота
        z=-30,      # высота (м)
        vx=0.0001,  # скорость по широте (градусов/с)
        vy=0.0001,  # скорость по долготе (градусов/с)
        vz=-1.0,    # вертикальная скорость (м/с)
        roll=0.0,   # крен (рад)
        pitch=0.0,  # тангаж (рад)
        heading=0.0  # курс (рад)
    )
    
    # Фактическая траектория (с погрешностями измерений)
    actual_state = AircraftState(
        x=55.5624 + np.random.normal(0, 0.0001),  # погрешность позиции
        y=37.9777 + np.random.normal(0, 0.0001),
        z=-30 + np.random.normal(0, 0.5),
        vx=0.0001 + np.random.normal(0, 0.00001),  # погрешность скорости
        vy=0.0001 + np.random.normal(0, 0.00001),
        vz=-1.0 + np.random.normal(0, 0.1),
        roll=0.0 + np.random.normal(0, 0.01),  # погрешность углов
        pitch=0.0 + np.random.normal(0, 0.01),
        heading=0.0 + np.random.normal(0, 0.01)
    )
    
    # Параметры неопределенности для расчета доверительных интервалов
    uncertainty = UncertaintyParams(
        pos_std=0.5,    # СКО позиции (м)
        vel_std=0.1,    # СКО скорости (м/с)
        att_std=0.01,   # СКО углов (рад)
        terrain_std=1.0  # СКО высоты рельефа (м)
    )
    
    # Расчет траекторий
    calculator = IntersectionCalculator(uncertainty)
    
    # Теоретическая траектория
    theoretical_dist = calculator.rk4_trajectory(
        theoretical_state, dt=0.1, steps=300
    )
    
    # Фактическая траектория
    actual_dist = calculator.rk4_trajectory(
        actual_state, dt=0.1, steps=300
    )
    
    # Расчет вероятностей пересечения
    theoretical_prob, _ = calculator.calculate_intersection_probability(
        theoretical_dist, terrain_points
    )
    actual_prob, _ = calculator.calculate_intersection_probability(
        actual_dist, terrain_points
    )
    
    # Визуализация результатов
    plot_results(
        theoretical_dist,
        actual_dist,
        terrain_points,
        theoretical_prob,
        actual_prob,
        uncertainty
    )
    
def plot_results(theoretical_dist, actual_dist, terrain_points, 
                theoretical_prob, actual_prob, uncertainty):
    """Визуализация результатов анализа"""
    
    # Создаем фигуру
    fig = plt.figure(figsize=(15, 10))
    
    # График траекторий
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Сравнение траекторий')
    
    # Отображаем рельеф
    terrain_x = [p.lat for p in terrain_points]
    terrain_y = [p.lon for p in terrain_points]
    terrain_z = [p.elevation for p in terrain_points]
    ax1.scatter(terrain_x, terrain_y, terrain_z, c='red', marker='^', s=100, label='Рельеф')
    
    # Извлекаем траектории
    theoretical_traj = np.array([dist[0] for dist in theoretical_dist])
    actual_traj = np.array([dist[0] for dist in actual_dist])
    
    # Отображаем теоретическую траекторию
    ax1.plot(
        theoretical_traj[:, 0],
        theoretical_traj[:, 1],
        -theoretical_traj[:, 2],
        color='blue',
        label='Теоретическая',
        linewidth=2
    )
    
    # Отображаем фактическую траекторию
    ax1.plot(
        actual_traj[:, 0],
        actual_traj[:, 1],
        -actual_traj[:, 2],
        color='green',
        label='Фактическая',
        linewidth=2
    )
    
    # Отображаем доверительные интервалы
    for j in range(0, len(actual_traj), 30):  # каждые 30 точек
        pos_mean = actual_traj[j]
        pos_cov = actual_dist[j][1]
        
        # Эллипсоид неопределенности
        U, S, _ = np.linalg.svd(pos_cov)
        radii = 2 * np.sqrt(S)  # 95% доверительный интервал
        
        # Создаем точки на поверхности эллипсоида
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        
        x = pos_mean[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
        y = pos_mean[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
        z = -pos_mean[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
        
        ax1.plot_surface(x, y, z, color='green', alpha=0.1)
    
    ax1.set_xlabel('Широта')
    ax1.set_ylabel('Долгота')
    ax1.set_zlabel('Высота (м)')
    ax1.legend()
    
    # График погрешностей
    ax2 = fig.add_subplot(122)
    ax2.set_title('Анализ погрешностей')
    
    # Расчет погрешностей
    pos_error = np.linalg.norm(theoretical_traj - actual_traj, axis=1)
    time_points = np.arange(len(pos_error)) * 0.1  # шаг 0.1с
    
    # Отображаем погрешность позиции
    ax2.plot(time_points, pos_error, label='Погрешность позиции', color='red')
    
    # Добавляем границы 3σ
    sigma = np.sqrt(uncertainty.pos_std**2 + uncertainty.vel_std**2 * time_points**2)
    ax2.fill_between(time_points, -3*sigma, 3*sigma, color='gray', alpha=0.2, label='3σ границы')
    
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Погрешность (м)')
    ax2.grid(True)
    ax2.legend()
    
    # Добавляем текстовую информацию
    info_text = (
        f'Параметры неопределенности:\n'
        f'σ_pos = {uncertainty.pos_std:.2f} м\n'
        f'σ_vel = {uncertainty.vel_std:.2f} м/с\n'
        f'σ_att = {uncertainty.att_std:.3f} рад\n'
        f'σ_terrain = {uncertainty.terrain_std:.1f} м\n\n'
        f'Вероятность пересечения:\n'
        f'Теоретическая: {theoretical_prob:.3f}\n'
        f'Фактическая: {actual_prob:.3f}'
    )
    
    plt.figtext(0.98, 0.98, info_text,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig('intersection_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    analyze_intersection_probability() 