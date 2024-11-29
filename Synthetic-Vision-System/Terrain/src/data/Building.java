// Пакет для организации классов
package data;

// Импорт необходимых классов
import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

// Импорт классов для работы с векторами из библиотеки LWJGL
import org.lwjgl.util.vector.Vector2f;
import org.lwjgl.util.vector.Vector3f;

// Импорт утилитного класса для триангуляции полигонов
import util.PolygonTriangulationUtil;

// Класс, представляющий здание в 3D пространстве
public class Building {

    // Высота здания
    private int height;
    // Список точек, определяющих углы основания здания
    private List<normPoint> corners;
    // Массив вершин для отрисовки здания
    public float[] vertices;
    // Массив индексов для определения порядка отрисовки треугольников
    public int[] indices;
    // Массив нормалей для освещения
    public float[] normals;
    
    // Конструктор здания
    public Building(int height, List<normPoint> corners) {
        setHeight(height);
        setCorners(corners);
        // Генерация массивов для отрисовки
        this.vertices = vecToArray(this.generateVertices());
        this.indices = intToArray(this.generateIndices());
        this.normals = vecToArray(this.generateVertexNormals());
    }
    
    // Геттеры и сеттеры для высоты
    public int getHeight() {
        return height;
    }
    public void setHeight(int height) {
        this.height = height;
    }
    
    // Геттеры и сеттеры для углов
    public List<normPoint> getCorners() {
        return corners;
    }
    public void setCorners(List<normPoint> corners) {
        this.corners = corners;
    }
    
    // Нормализация вектора (приведение к единичной длине)
    public static Vector3f normalize(Vector3f init) {
        float x = init.getX();
        float y = init.getY();
        float z = init.getZ();
        
        // Вычисление длины вектора
        float squaredsum = (x * x) + (y * y) + (z * z);
        float magnitude = (float) Math.sqrt(squaredsum);
        
        // Создание нормализованного вектора
        return new Vector3f(x / magnitude, y / magnitude, z / magnitude);
    }
    
    // Преобразование списка векторов в массив float для OpenGL
    public static float[] vecToArray(List<Vector3f> listOfVectors) {
        List<Float> temp = new ArrayList<Float>();
        float[] ret = new float[listOfVectors.size() * 3];
        
        // Разбор каждого вектора на компоненты
        for (int i = 0; i < listOfVectors.size(); i++) {
            Vector3f vec = listOfVectors.get(i);
            temp.add(vec.getX());
            temp.add(vec.getY());
            temp.add(vec.getZ());
        }
        
        // Копирование во float массив
        for (int i = 0; i < temp.size(); i++) {
            ret[i] = temp.get(i);
        }
        return ret;
    }
    
    // Преобразование списка целых чисел в массив
    public static int[] intToArray(List<Integer> listOfInts) {
        int[] ret = new int[listOfInts.size()];
        
        for (int i = 0; i < listOfInts.size(); i++) {
            ret[i] = listOfInts.get(i);
        }
        return ret;
    }
    
    // Генерация вершин для 3D модели здания
    public List<Vector3f> generateVertices() {
        List<Vector3f> vec = new ArrayList<Vector3f>();
        normPoint init = this.corners.get(0);
        
        // Реальная высота (масштабированная)
        float realHeight = 3 * this.height;
        
        // Генерация боковых граней
        vec.add(new Vector3f(init.getX(), 0, init.getY()));
        vec.add(new Vector3f(init.getY(), realHeight, init.getY()));
        
        // Создание вершин для каждого угла
        for (int i = 1; i < this.corners.size(); i++) {
            normPoint corner = this.corners.get(i);
            vec.add(new Vector3f(corner.getX(), 0, corner.getY()));
            vec.add(new Vector3f(corner.getX(), realHeight, corner.getY()));
            
            vec.add(new Vector3f(corner.getX(), 0, corner.getY()));
            vec.add(new Vector3f(corner.getX(), realHeight, corner.getY()));
        }
        
        // Замыкание боковых граней
        vec.add(new Vector3f(init.getX(), 0, init.getY()));
        vec.add(new Vector3f(init.getY(), realHeight, init.getY()));
        
        // Добавление вершин крыши
        for (normPoint point: this.corners) {
            vec.add(new Vector3f(point.getX(), realHeight, point.getY()));
        }
        
        return vec;
    }
    
    // Генерация индексов для построения треугольников
    public List<Integer> generateIndices() {
        List<Integer> indices = new ArrayList<Integer>();
        
        // Индексы для боковых граней
        for (int i = 1; i < this.corners.size() - 1; i++) {
            int iterate = i * 4;
            indices.add(iterate + 1);
            indices.add(iterate);
            indices.add(iterate + 3);
            indices.add(iterate);
            indices.add(iterate + 2);
            indices.add(iterate + 3);
        }
        
        // Преобразование точек для триангуляции крыши
        List<Point> truePoints = new ArrayList<Point>();
        for (normPoint p: this.corners) {
            Point n = new Point();
            n.setLocation(p.getX(), p.getY());
            truePoints.add(n);
        }
        
        // Получение индексов для крыши
        List<Integer> roofIndices = PolygonTriangulationUtil.getPolygonTriangulationIndices(truePoints, true);
        for (int i = 0; i < roofIndices.size(); i++) {
            roofIndices.set(i, roofIndices.get(i) + ((4 * this.corners.size())));
        }
        indices.addAll(roofIndices);
        return indices;
    }
    
    // Преобразование вектора в положительные значения
    public static Vector3f posify(Vector3f init) {
        return new Vector3f(Math.abs(init.getX()), Math.abs(init.getY()), Math.abs(init.getZ()));
    }
    
    // Генерация нормалей для освещения
    public List<Vector3f> generateVertexNormals() {
        List<Vector3f> normals = new ArrayList<Vector3f>();
        List<Vector3f> verts = this.generateVertices();
        List<Integer> inds = this.generateIndices();
        
        // Инициализация нормалей нулевыми векторами
        for (int i = 0; i < verts.size(); i++) {
            normals.add(new Vector3f(0, 0, 0));
        }
        
        // Вычисление нормалей для каждого треугольника
        for (int i = 0; i < inds.size() - 1; i +=3) {
            int vertA = inds.get(i);
            int vertB = inds.get(i + 1);
            int vertC = inds.get(i + 2);
            
            Vector3f posA = verts.get(vertA);
            Vector3f posB = verts.get(vertB);
            Vector3f posC = verts.get(vertC);
            
            // Вычисление векторов граней
            Vector3f edgeAB = Vector3f.sub(posA, posB, null);
            Vector3f edgeAC = Vector3f.sub(posA, posC, null);
            
            // Вычисление нормали как векторного произведения
            Vector3f cross = Vector3f.cross(edgeAB, edgeAC, null);
            
            cross = normalize(cross);
            Vector3f newCross = new Vector3f(cross.getX(), -1 * cross.getY(), -1 * cross.getZ());
            normals.set(vertA, newCross);
            normals.set(vertB, newCross);
            normals.set(vertC, newCross);
        }
        
        // Обработка нормалей для крыши
        for (int i = normals.size() - this.corners.size(); i < normals.size(); i++) {
            normals.set(i, posify(normals.get(i)));
        }
        
        return normals;
    }
    
    // Строковое представление объекта
    public String toString() {
        String s = "";
        s += "Height: " + this.height;
        s += ", Corners: [";
        
        for (normPoint p: this.corners) {
            s += p;
            s += ", ";
        }
        
        s += "]";
        
        return s;
    }
}
