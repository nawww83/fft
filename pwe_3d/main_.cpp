#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

using namespace std;
using Complex = complex<double>;

const Complex I(0.0, 1.0);

struct GridConfig {
    int Ny;           // Точек по Y
    int Nz;           // Точек по Z
    double dy;        // Шаг по Y
    double dz;        // Шаг по Z
    double dx;        // Шаг по X
    double k0;        // Волновое число в вакууме
    Complex eps_g;    // Комплексная проницаемость почвы: eps_r + i * sigma / (omega * eps_0)
    int pml_width;    // Ширина поглощающего слоя в количестве ячеек
    double pml_max;   // Максимальная сила поглощения на самом краю
};

struct Terrain {
    double h;
    double hy;
    double hyy;
};

// Пример рельефа
Terrain get_terrain(double x, double y) {
    Terrain t;
    double omega = 0.05; 
    t.h = 20.0 * sin(omega * y);
    t.hy = 20.0 * omega * cos(omega * y);
    t.hyy = -20.0 * omega * omega * sin(omega * y);
    return t;
}

// Функция для вычисления профиля поглощения PML
Complex get_pml_factor(int j, int m, const GridConfig& cfg) {
    double sigma = 0.0;
    
    // Поглощение у левой границы
    if (j < cfg.pml_width) {
        double d = double(cfg.pml_width - j) / cfg.pml_width;
        sigma += cfg.pml_max * (d * d * d); // Кубический закон нарастания
    }
    // Поглощение у правой границы
    if (j >= cfg.Ny - cfg.pml_width) {
        double d = double(j - (cfg.Ny - cfg.pml_width - 1)) / cfg.pml_width;
        sigma += cfg.pml_max * (d * d * d);
    }
    // Поглощение у верхней границы
    if (m >= cfg.Nz - cfg.pml_width) {
        double d = double(m - (cfg.Nz - cfg.pml_width - 1)) / cfg.pml_width;
        sigma += cfg.pml_max * (d * d * d);
    }

    // Возвращаем затухание, которое добавится как мнимая часть волнового числа
    // u ~ exp(i * (k0 + i*sigma) * x) -> exp(-sigma * x)
    return -I * sigma;
}

void make_pe_step_advanced(const GridConfig& cfg, double current_x, 
                           const vector<vector<Complex>>& u_current, 
                           vector<vector<Complex>>& u_next) 
{
    int Ny = cfg.Ny;
    int Nz = cfg.Nz;
    double dy = cfg.dy;
    double dz = cfg.dz;
    double dx = cfg.dx;
    double k0 = cfg.k0;

    Complex alpha = dx / (4.0 * I * k0);
    const int MAX_ITER = 150;
    const double TOLERANCE = 1e-5;
    const double omega_sor = 1.6; 

    u_next = u_current;

    vector<Terrain> terr(Ny);
    for (int j = 0; j < Ny; ++j) {
        terr[j] = get_terrain(current_x + dx/2.0, j * dy);
    }

    // Расчет коэффициента Леонтовича для горизонтальной поляризации (как пример)
    // eta = 1.0 / sqrt(eps_g - 1.0)
    Complex eta = 1.0 / sqrt(cfg.eps_g - 1.0);

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double max_residual = 0.0;

        // 1. СНАЧАЛА ОБНОВЛЯЕМ ВНУТРЕННИЕ ТОЧКИ (с учетом PML)
        for (int j = 1; j < Ny - 1; ++j) {
            for (int m = 1; m < Nz - 1; ++m) {
                
                double hy = terr[j].hy;
                double hyy = terr[j].hyy;
                Complex pml = get_pml_factor(j, m, cfg);

                // Конечные разности на текущем итерационном слое (next)
                Complex d2u_dy2_next = (u_next[j+1][m] - 2.0*u_next[j][m] + u_next[j-1][m]) / (dy * dy);
                Complex d2u_dz2_next = (u_next[j][m+1] - 2.0*u_next[j][m] + u_next[j][m-1]) / (dz * dz);
                Complex du_dz_next   = (u_next[j][m+1] - u_next[j][m-1]) / (2.0 * dz);
                Complex d2u_dydz_next = (u_next[j+1][m+1] - u_next[j+1][m-1] - u_next[j-1][m+1] + u_next[j-1][m-1]) / (4.0 * dy * dz);

                Complex L_next_explicit = d2u_dy2_next + (1.0 + hy*hy)*d2u_dz2_next - 2.0*hy*d2u_dydz_next - hyy*du_dz_next;
                Complex center_coeff_next = -2.0/(dy*dy) - 2.0*(1.0 + hy*hy)/(dz*dz);

                // Разности на предыдущем шаге (current)
                Complex d2u_dy2_curr = (u_current[j+1][m] - 2.0*u_current[j][m] + u_current[j-1][m]) / (dy * dy);
                Complex d2u_dz2_curr = (u_current[j][m+1] - 2.0*u_current[j][m] + u_current[j][m-1]) / (dz * dz);
                Complex du_dz_curr   = (u_current[j][m+1] - u_current[j][m-1]) / (2.0 * dz);
                Complex d2u_dydz_curr = (u_current[j+1][m+1] - u_current[j+1][m-1] - u_current[j-1][m+1] + u_current[j-1][m-1]) / (4.0 * dy * dz);
                
                Complex L_curr = d2u_dy2_curr + (1.0 + hy*hy)*d2u_dz2_curr - 2.0*hy*d2u_dydz_curr - hyy*du_dz_curr;

                // Добавляем влияние PML: член k0^2 * (n^2 - 1) превращается в эффективный поглотитель 2 * k0 * pml
                Complex pml_term = 2.0 * k0 * pml;

                // Уравнение Кранка-Николсона с PML
                Complex RHS = u_current[j][m] + alpha * (L_curr + pml_term * u_current[j][m]) 
                                              + alpha * (L_next_explicit - center_coeff_next * u_next[j][m]);
                
                Complex denominator = 1.0 - alpha * (center_coeff_next + pml_term);
                Complex u_next_target = RHS / denominator;

                Complex delta = u_next_target - u_next[j][m];
                u_next[j][m] += omega_sor * delta;
                max_residual = max(max_residual, abs(delta));
            }
        }

        // 2. ОБНОВЛЯЕМ НИЖНЮЮ ГРАНИЦУ (m = 0, Леонтович на рельефе)
        // В декартовых ГУ: du/dz + ik0*eta*u = 0.
        // В выпрямленных координатах с учетом уклона: du/dz' + ik0*eta*sqrt(1 + hy^2)*u = 0
        // Аппроксимируем du/dz' односторонней разностью второго порядка точности по точкам m=0,1,2:
        // du/dz' = (-3*u[0] + 4*u[1] - u[2]) / (2*dz)
        for (int j = 1; j < Ny - 1; ++j) {
            double hy = terr[j].hy;
            Complex BC_coeff = I * k0 * eta * sqrt(1.0 + hy * hy);
            
            // Выражаем u_next[j][0] через внутренние точки m=1 и m=2
            Complex u_bottom_target = (4.0 * u_next[j][1] - u_next[j][2]) / (3.0 + 2.0 * dz * BC_coeff);
            
            Complex delta = u_bottom_target - u_next[j][0];
            u_next[j][0] += omega_sor * delta; 
            max_residual = max(max_residual, abs(delta));
        }

        // 3. ЖЕСТКИЕ КРАЕВЫЕ УСЛОВИЯ (Самый край PML сетки зануляем)
        for (int m = 0; m < Nz; ++m) {
            u_next[0][m] = 0.0;
            u_next[Ny-1][m] = 0.0;
        }
        for (int j = 0; j < Ny; ++j) {
            u_next[j][Nz-1] = 0.0;
        }

        if (max_residual < TOLERANCE) break;
    }
}

int main() {
    GridConfig cfg;
    cfg.Ny = 120;             // Увеличили сетку, чтобы влез PML по бокам
    cfg.Nz = 70;              // Увеличили сетку для PML сверху
    cfg.dy = 0.2;
    cfg.dz = 0.1;
    cfg.dx = 0.4;
    cfg.k0 = 2.0 * M_PI / 1.0; 
    
    // Параметры почвы (влажная земля на СВЧ, например eps = 15, проводимость дает мнимую часть)
    cfg.eps_g = Complex(15.0, 2.0); 
    
    // Настройки PML
    cfg.pml_width = 15;       // 15 ячеек с каждого края гасят волну
    cfg.pml_max = 5.0;        // Коэффициент затухания

    vector<vector<Complex>> u_current(cfg.Ny, vector<Complex>(cfg.Nz, 0.0));
    vector<vector<Complex>> u_next(cfg.Ny, vector<Complex>(cfg.Nz, 0.0));

    // Инициализация источника (чуть выше рельефа и за пределами левого PML)
    u_current[cfg.pml_width + 10][5] = 1.0; 

    double x = 0.0;
    double max_distance = 5.0;

    while (x < max_distance) {
        make_pe_step_advanced(cfg, x, u_current, u_next);
        u_current = u_next;
        x += cfg.dx;
    }

    cout << "Расчет с физическими границами успешно завершен!" << endl;
    return 0;
}


//  Как это работает с физической точки зрения:
//  Модифицированный Леонтович:
//  Когда земля идет под наклоном, нормаль к поверхности больше не совпадает с вертикальной осью \(z\). 
//  Математически это приводит к тому, что стандартный импеданс умножается на геометрический фактор наклона: 
//  \(\sqrt{1+h_{y}^{2}}\).
//  В коде использована односторонняя разность второго порядка (-3*u[0] + 4*u[1] - u[2]), что сохраняет 
//  общую точность схемы Кранка-Николсона \(O(\Delta z^2)\).
//  Интеграция PML во внутренний цикл:
//  Мы не просто зануляем края. Мы добавили плавный поглотитель. Функция get_pml_factor возвращает затухание, 
//  которое нарастает кубически (d*d*d). Это гарантирует, что волна, доходя до боковых стен или верха расчетной 
//  зоны, «вязнет» в них и её амплитуда падает до нуля еще до того, как она коснется жестких границ u = 0.

//  В коде для простоты был использован метод эффективной мнимой проводимости (так называемый Sponge layer), где 
//  мы просто добавили мнимую часть к волновому числу: \(k_0 \to k_0 - i\sigma\). Для параболического уравнения, где 
//  волна идет под узкими углами вперед, этот метод при плавной кубической форме нарастания \(\sigma \) дает отличный 
//  результат, почти неотличимый от строгого PML.

//  Теоретически (на бумаге, в непрерывных дифференциальных уравнениях) отражение от PML строго равно нулю. 
//  Однако, как только мы переходим на дискретную сетку в компьютере, возникают две проблемы:
//  Дискретизация: Мы не можем сделать шаг \(\Delta x \to 0\). Дискретный скачок коэффициента \(s_{x}\) от узла к узлу 
//  все-таки порождает крошечное численное отражение.
//  Углы падения: Если волна падает на границу PML почти параллельно ей (скользящее падение), численное отражение 
//  резко возрастает.Чтобы с этим бороться, профиль проводимости \(\sigma(s)\) никогда не делают ступенькой. 
//  Его делают плавно нарастающим к краю расчетной области по параболическому или кубическому закону 
//  (\(\sigma(s) = \sigma_{max} \cdot (s / L)^3\)), как мы и поступили в нашей C++ программе.


// В разработанном нами алгоритме мы учли полное трехмерное узкоугольное параболическое уравнение (3D PE), 
// адаптированное под искривленную систему координат.
// Если записать физическое уравнение, которое фактически зашито в наш численный шаг схемы Кранка-Николсона, 
// оно состоит из следующих членов:
// \(2ik_{0}\frac{\partial u}{\partial x}+\frac{\partial ^{2}u}{\partial y^{\prime 2}}+(1+h_{y}^{2})\frac{\partial ^{2}u}{\partial z^{\prime 2}}-2h_{y}\frac{\partial ^{2}u}{\partial y^{\prime }\partial z^{\prime }}-h_{yy}\frac{\partial u}{\partial z^{\prime }}+2k_{0}^{2}\cdot \Delta n_{\text{pml}}\cdot u=0\)
// Давайте разберем каждый член, который мы перенесли в код, и за что он отвечает:
// 1. Маршевый член по оси распространения
// Математический вид: 
// \(2ik_0 \frac{\partial u}{\partial x}\)
// В коде: Представлен разностью между слоями (u_next - u_current) / dx с множителем alpha.
// Физический смысл: Отвечает за движение волнового фронта вперед. Из-за него метод работает как «видеопоток», рассчитывая плоскость за плоскостью.
// 2. Поперечная диффузия (Растекание вбок)
// Математический вид: 
// \(\frac{\partial ^{2}u}{\partial y^{\prime 2}}\)
// В коде: Переменные d2u_dy2_next и d2u_dy2_curr.
// Физический смысл: Описывает классическую дифракцию — пучок света или радиоволн естественным образом расширяется в горизонтальной плоскости по мере 
// удаления от антенны.
// 3. Вертикальная диффузия с поправкой на наклон рельефа
// Математический вид: 
// \((1 + h_y^2) \frac{\partial^2 u}{\partial z'^2}\)
// В коде: Коэффициент (1.0 + hy*hy) перед вертикальными вторыми производными d2u_dz2.
// Физический смысл: Описывает расширение пучка вверх и вниз. Множитель \(h_{y}^{2}\) корректирует этот процесс, если подстилающая поверхность под углом 
// уходит вбок (склон холма), компенсируя искажение сетки.
// 4. Смешанная производная (Главная сложность сильного рельефа)
// Математический вид: \(- 2h_y \frac{\partial^2 u}{\partial y' \partial z'}\)
// В коде: Член с перекрестной центральной разностью по 4-м точкам d2u_dydz_next и d2u_dydz_curr с множителем -2.0 * hy.
// Физический смысл: Учитывает перекос волнового фронта. Если волна идет вдоль косого склона, энергия начинает «стекать» по диагонали сетки. 
// Именно из-за этого члена матрица перестала быть трехдиагональной, и нам пришлось внедрить итерационный цикл SOR.
// 5. Эффективный снос («Ветер» рельефа)
// Математический вид: \(- h_{yy} \frac{\partial u}{\partial z'}\)
// В коде: Первая производная по вертикали du_dz с коэффициентом -hyy (вторая производная рельефа).
// Физический смысл: Этот член работает как локальное искривление пространства. Когда рельеф резко изгибается (выпуклая вершина холма или вогнутое дно оврага, \(h_{yy} \neq 0\)), 
// этот член принудительно наклоняет фазовый фронт волны, заставляя её огибать препятствие.
// 6. Член искусственного поглощения (PML / Sponge)
// Математический вид: \(2k_0^2 \cdot \Delta n_{\text{pml}} \cdot u\), 
// где \(\Delta n_{\text{pml}} = -i \frac{\sigma(y,z)}{k_0}\)
// В коде: Переменная pml_term, рассчитываемая через функцию get_pml_factor().
// Физический смысл: Имитирует появление сильного затухания в воздухе возле верхней и боковых границ. 
// Он превращает экспоненту несущей волны из колеблющейся в затухающую, что убирает отражения от краев.
// 💡 Что мы НЕ учли (и почему это оправдано)?
// Вторую производную по рельефу вдоль хода луча (\(h_{xx}\)): 
// Мы посчитали, что рельеф вдоль оси \(x\) меняется достаточно плавно. На малых углах (\(5^\circ - 25^\circ\)) влияние \(h_{xx}\) ничтожно мало.
// Обратное рассеяние: Метод параболического уравнения принципиально игнорирует волны, отраженные назад к антенне. 
// Мы считаем только поле, летящее строго вперед.Если в вашей задаче планируется учитывать атмосферную рефракцию 
// (когда коэффициент преломления воздуха \(n\) меняется с высотой, например, из-за миражей или температурных инверсий), 
// этот эффект добавляется в самый последний член как реальная часть: \(+k_0^2(n^2(x,y,z) - 1)u\).
