#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;
using Complex = complex<double>;
const Complex I(0.0, 1.0);

struct GridConfig {
    int Ny;
    int Nz;
    double dy;
    double dz;
    double k0;
    int pml_width;
};

// 1. Модель диаграммы направленности (ДН) антенны по полю
// Возвращает комплексную амплитуду излучения под углами theta_y и theta_z (в радианах)
Complex antenna_pattern(double theta_y, double theta_z) {
    // Пример: Гауссов пучок, наклоненный вверх под углом 15 градусов (0.26 rad)
    // и имеющий ширину (диаграмму) около 10 градусов (0.17 rad)
    double target_theta_z = 15.0 * M_PI / 180.0; // Центр луча по вертикали
    double target_theta_y = 0.0;                 // Центр луча по горизонтали
    
    double beam_width_y = 8.0 * M_PI / 180.0;
    double beam_width_z = 8.0 * M_PI / 180.0;

    double dy = (theta_y - target_theta_y) / beam_width_y;
    double dz = (theta_z - target_theta_z) / beam_width_z;

    // Возвращаем амплитуду ДН (Гауссоида). Здесь же можно задать фазовое распределение, если нужно
    return exp(-(dy * dy + dz * dz));
}

// 2. Функция инициализации поля на плоскости X = 0
void initialize_antenna_field(const GridConfig& cfg, vector<vector<Complex>>& u_start, double antenna_z_center) {
    int Ny = cfg.Ny;
    int Nz = cfg.Nz;
    double dy = cfg.dy;
    double dz = cfg.dz;
    double k0 = cfg.k0;

    // Временная матрица для спектра поля в k-пространстве
    vector<vector<Complex>> u_spectrum(Ny, vector<Complex>(Nz, 0.0));

    // Шаги по пространственным частотам (спектральная сетка)
    double dky = (2.0 * M_PI) / (Ny * dy);
    double dkz = (2.0 * M_PI) / (Nz * dz);

    // Заполняем спектр значениями ДН антенны
    for (int j = 0; j < Ny; ++j) {
        // Преобразуем индекс в циклическую частоту ky (учитывая отрицательные частоты)
        double ky = (j < Ny / 2) ? j * dky : (j - Ny) * dky;
        double theta_y = asin(ky / k0); // Угол распространения по Y

        for (int m = 0; m < Nz; ++m) {
            double kz = (m < Nz / 2) ? m * dkz : (m - Nz) * dkz;
            double theta_z = asin(kz / k0); // Угол распространения по Z

            // Проверяем, вещественны ли углы (избегаем запредельных волн в ДН)
            if (abs(ky / k0) <= 1.0 && abs(kz / k0) <= 1.0) {
                u_spectrum[j][m] = antenna_pattern(theta_y, theta_z);
            } else {
                u_spectrum[j][m] = 0.0;
            }
        }
    }

    // Выполняем Двумерное Обратное Дискретное Преобразование Фурье (2D IDFT)
    // Чтобы перевести спектр в распределение поля по координатам y и z
    for (int y_idx = 0; y_idx < Ny; ++y_idx) {
        double y = y_idx * dy;
        for (int z_idx = 0; z_idx < Nz; ++z_idx) {
            
            // Физическая координата Z (высота). 
            // Субъективно смещаем фазовый центр антенны на высоту antenna_z_center над рельефом
            double z = z_idx * dz - antenna_z_center; 

            Complex sum = 0.0;
            for (int j = 0; j < Ny; ++j) {
                double ky = (j < Ny / 2) ? j * dky : (j - Ny) * dky;
                for (int m = 0; m < Nz; ++m) {
                    double kz = (m < Nz / 2) ? m * dkz : (m - Nz) * dkz;

                    // Экспонента обратного преобразования Фурье
                    Complex phase = I * (ky * y + kz * z);
                    sum += u_spectrum[j][m] * exp(phase);
                }
            }
            
            // Нормировка Фурье и запись в стартовый слой поля
            u_start[y_idx][z_idx] = sum / double(Ny * Nz);

            // Плавное обнуление поля внутри зон PML на старте, чтобы не было паразитных вспышек
            if (y_idx < cfg.pml_width || y_idx >= Ny - cfg.pml_width || z_idx >= Nz - cfg.pml_width) {
                u_start[y_idx][z_idx] = 0.0;
            }
        }
    }
}

int main() {
    GridConfig cfg;
    cfg.Ny = 64;             // Для быстроты демонстрационного DFT размер сетки уменьшен
    cfg.Nz = 32;
    cfg.dy = 0.2;            // Шаг сетки должен быть меньше длины волны (например, lambda/5)
    cfg.dz = 0.2;
    cfg.k0 = 2.0 * M_PI / 1.0; // Длина волны lambda = 1.0 метр
    cfg.pml_width = 10;

    vector<vector<Complex>> u_initial(cfg.Ny, vector<Complex>(cfg.Nz, 0.0));

    // Размещаем фазовый центр антенны на высоте 3.0 метра над рельефом
    double antenna_height = 3.0; 
    
    initialize_antenna_field(cfg, u_initial, antenna_height);

    cout << "Поле антенны успешно сформировано из ДН на плоскости X=0!" << endl;
    cout << "Амплитуда в центре расчетной области: " << abs(u_initial[cfg.Ny/2][15]) << endl;

    return 0;
}

// Важные инженерные нюансы интеграции:
// Фазовый центр (Высота antenna_z_center): 
// В коде координата z при вычислении экспоненты Фурье сдвигается на величину высоты подвеса антенны. 
// Это автоматически закладывает правильный наклон фазового фронта (сферичность волны) относительно земли, 
// как если бы антенна физически висела на этой высоте.
// Шаг сетки и дискретизация углов: Чтобы корректно отобразить углы до \(25^{\circ }\), шаги сетки
//  \(\Delta y\) и \(\Delta z\) должны удовлетворять критерию Котельникова-Шеннона для максимальной 
//  пространственной частоты. На практике выбирайте шаг не более \(\lambda / 4\)... \(\lambda / 6\). 
//  Например, для \(\lambda = 1\) м, шаги \(dy, dz = 0.15...0.2\) м будут идеальны.
//  Оптимизация (FFT): Если вы решите увеличить сетку до реальных размеров (например, \(512 \times 512\)), 
//  вложенный четырехкратный цикл DFT в функции инициализации начнет заметно тормозить. В этом случае замените 
//  этот блок на вызов стандартной функции 2D IFFT из любой готовой библиотеки (например, pocketfft или FFTW). 
//  Принцип заполнения матрицы частот u_spectrum останется точно таким же.

// Математический принцип
// Пусть задана ДН антенны по полю как функция углов: \(F(\theta, \phi)\), где 
// \(\theta \) — угол к оси распространения 
// \(x\) (элевация), а 
// \(\phi \) — азимут в плоскости \(y-z\).
// В методе параболического уравнения мы работаем в пространстве спектральных частот (волновых векторов) \(k_{y}\) и \(k_{z}\). 
// Направляющие косинусы углов связаны с компонентами волнового вектора следующим образом:
// \(k_{y}=k_{0}\cdot \sin (\theta )\cos (\phi )\)\(k_{z}=k_{0}\cdot \sin (\theta )\sin (\phi )\)
// Если у вас углы распространения малые (\(5^\circ - 25^\circ\)), то можно использовать приближение малых углов, 
// где \(k_y \approx k_0 \cdot \theta_y\) и \(k_z \approx k_0 \cdot \theta_z\).
// Алгоритм заполнения стартовой плоскости:
// Создаем матрицу спектра такого же размера, как наша сетка: U_spectrum[Ny][Nz].
// Каждому индексу частоты \((k_y, k_z)\) сопоставляем физический угол излучения.
// Записываем в ячейку значение ДН для этого угла.
// Выполняем обратное преобразование Фурье (IDFT), чтобы перевести поле из углового спектра в физические координаты \((y, z)\).