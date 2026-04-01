import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.ticker import ScalarFormatter

# 1. ЗАГРУЗКА ДАННЫХ ИЗ ФАЙЛА
try:
    df = pd.read_csv('results.csv')
except FileNotFoundError:
    print("Ошибка: Файл 'results.csv' не найден.")
    exit(1)

# 2. ПАРАМЕТРЫ СРЕДЫ
CPU_FREQ_GHZ = 1.99
COMPILER = "MSVC 19.50"
FLAGS = "/Ox /fp:fast /arch:AVX2"
ELEM_SIZE = 16  # complex<double> = 16 байт

# 3. РАСЧЕТ МЕТРИК
def calculate_metrics(row):
    n = row['N']
    total_ops = (10.0 * n * np.log2(n)) + n
    m_mean = (total_ops / (row['Mean'] / 1000.0)) / 1e6
    m_low = (total_ops / ((row['Mean'] + row['CI95']) / 1000.0)) / 1e6
    m_high = (total_ops / ((row['Mean'] - row['CI95']) / 1000.0)) / 1e6
    eff = total_ops / (row['Mean'] / 1000.0 * CPU_FREQ_GHZ * 1e9)
    return pd.Series([m_mean, m_low, m_high, eff])

df[['M_MEAN', 'M_LOW', 'M_HIGH', 'EFF']] = df.apply(calculate_metrics, axis=1)

# 4. ПОСТРОЕНИЕ ГРАФИКА
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx() 

for algo in df['Algorithm'].unique():
    d = df[df['Algorithm'] == algo].sort_values('N')
    
    line, = ax1.plot(d['N'], d['M_MEAN'], marker='o', linewidth=2, label=f'{algo} MFLOPS')
    color = line.get_color()
    ax1.fill_between(d['N'], d['M_LOW'], d['M_HIGH'], color=color, alpha=0.15)
    ax2.plot(d['N'], d['SNR'], linestyle='--', marker='x', alpha=0.4, color=color, label=f'{algo} SNR')
    
    # Аннотации эффективности (FLOPS/Cycle)
    for _, row in d.iterrows():
        ax1.text(row['N'], row['M_MEAN'] + (df['M_MEAN'].max() * 0.03), f"{row['EFF']:.2f}", 
                ha='center', fontsize=8, color=color, fontweight='bold')

# 5. ГРАНИЦЫ КЭША (сдвинуты чуть правее линии)
cache_info = [
    ("L1 (32 KB)", 32 * 1024),
    ("L2 (256 KB)", 256 * 1024),
    ("L3 (8 MB)", 8 * 1024 * 1024)
]

y_lims = ax1.get_ylim()
for name, size_bytes in cache_info:
    n_limit = size_bytes // ELEM_SIZE
    ax1.axvline(x=n_limit, color='red', linestyle=':', alpha=0.4)
    
    # Сдвиг вправо: умножаем координату x на 1.1 (т.к. шкала логарифмическая)
    label_x = n_limit * 1.1 
    label = f"{name} ≈ {n_limit if n_limit < 1000 else f'{n_limit//1024}K'} el."
    ax1.text(label_x, ax1.get_ylim()[0] * 1.05, label, 
             color='red', alpha=0.6, rotation=90, va='bottom', fontsize=9)

# Зона RAM Bound
ax1.axvspan(524288, df['N'].max(), color='gray', alpha=0.05)

# 6. НАСТРОЙКА ОСЕЙ
ax1.set_xscale('log', base=2)
ticks = sorted(df['N'].unique())
ax1.set_xticks(ticks)

formatter = ScalarFormatter()
formatter.set_scientific(False)
ax1.xaxis.set_major_formatter(formatter)
plt.setp(ax1.get_xticklabels(), rotation=45)

ax1.set_ylabel('Производительность (MFLOPS)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Точность (SNR dB)', fontsize=11, color='gray')
ax1.set_xlabel('Размер данных N', fontsize=11)
ax1.set_title('Бенчмарк БПФ: Производительность (FP64) и Точность (SNR)', fontsize=14, pad=20)

# Компактные границы
ax1.set_ylim(df['M_LOW'].min() * 0.8, df['M_HIGH'].max() * 1.3)
ax2.set_ylim(df['SNR'].min() - 2, df['SNR'].max() + 2)

# 7. ЛЕГЕНДА
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', shadow=True)

# 8. ИНФО-БЛОКИ
info_box = (f"{COMPILER} | {FLAGS}\n"
            f"CPU: i7-8565U @ {CPU_FREQ_GHZ}GHz\n"
            f"Date: {datetime.now().strftime('%d.%m.%Y')}")

ax1.text(0.02, 0.97, info_box, transform=ax1.transAxes, fontsize=8, family='monospace',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

desc_text = ("Числа над точками — FLOPS/Cycle\n(Операций FP64 за такт)")
ax1.text(0.02, 0.85, desc_text, transform=ax1.transAxes, fontsize=8, color='#333',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f9f9f9', alpha=0.9))

ax1.grid(True, which="major", axis='both', alpha=0.2, linestyle='--')

# 9. СОХРАНЕНИЕ
plt.savefig('fft_benchmark.png', dpi=300, bbox_inches='tight')
plt.show()
