import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

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

# 4. СТИЛИ: 4 семейства алгоритмов x {базовый, SIMD}
FAMILIES = ['Iterative AoS', 'Iterative SoA', 'Recursive AoS', 'Recursive SoA']
COLORS = {
    'Iterative AoS': '#1f77b4',
    'Iterative SoA': '#ff7f0e',
    'Recursive AoS': '#2ca02c',
    'Recursive SoA': '#d62728',
}
STYLE_BASE = dict(linestyle='--', linewidth=1.6, marker='o', markersize=6,
                  markerfacecolor='white', markeredgewidth=1.4, alpha=0.85)
STYLE_SIMD = dict(linestyle='-', linewidth=2.4, marker='o', markersize=7,
                  markeredgewidth=1.0, alpha=1.0)

# 5. ФИГУРА: сверху производительность, снизу точность
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(15, 10.5),
    gridspec_kw={'height_ratios': [3.0, 1.15], 'hspace': 0.08})
ax1b = ax1.twinx()  # правая ось: FLOPS/cycle

for fam in FAMILIES:
    c = COLORS[fam]
    for suffix, style in (('', STYLE_BASE), (' SIMD', STYLE_SIMD)):
        name = fam + suffix
        d = df[df['Algorithm'] == name].sort_values('N')
        if d.empty:
            continue
        st = dict(style)
        st.setdefault('markerfacecolor', c)
        ax1.plot(d['N'], d['M_MEAN'], color=c, label=name, **st)
        ax2.plot(d['N'], d['SNR'], color=c, **st)
        # Доверительный интервал показываем только у SIMD, чтобы не зашумлять
        if suffix:
            ax1.fill_between(d['N'], d['M_LOW'], d['M_HIGH'], color=c, alpha=0.13)

# 6. ГРАНИЦЫ КЭША
cache_info = [
    ("L1 (32 KB)", 32 * 1024),
    ("L2 (256 KB)", 256 * 1024),
    ("L3 (8 MB)", 8 * 1024 * 1024),
]
for name, size_bytes in cache_info:
    n_limit = size_bytes // ELEM_SIZE
    for ax in (ax1, ax2):
        ax.axvline(x=n_limit, color='red', linestyle=':', alpha=0.4)
    label = f"{name} ≈ {n_limit if n_limit < 1000 else f'{n_limit // 1024}K'} el."
    ax1.text(n_limit * 1.1, ax1.get_ylim()[0] * 1.05, label,
             color='red', alpha=0.6, rotation=90, va='bottom', fontsize=9)

# Зона RAM Bound
for ax in (ax1, ax2):
    ax.axvspan(524288, df['N'].max(), color='gray', alpha=0.05)

# 7. НАСТРОЙКА ОСЕЙ
ax1.set_xscale('log', base=2)
ticks = sorted(df['N'].unique())
ax1.set_xticks(ticks)
formatter = ScalarFormatter()
formatter.set_scientific(False)
ax1.xaxis.set_major_formatter(formatter)
plt.setp(ax2.get_xticklabels(), rotation=45)

ax1.set_ylabel('Производительность (MFLOPS)', fontsize=11, fontweight='bold')
ax1.set_ylim(df['M_LOW'].min() * 0.8, df['M_HIGH'].max() * 1.32)
ax1.set_xlabel('')

ax2.set_ylabel('Точность (SNR dB)', fontsize=11, color='gray')
ax2.set_xlabel('Размер данных N', fontsize=11)
ax2.set_title('Точность относительно эталона (все реализации > 300 дБ)',
              fontsize=10, color='gray', pad=6)

# Правая ось — FLOPS/cycle (линейный масштаб от MFLOPS: EFF = MFLOPS / (GHz * 1000))
lo, hi = ax1.get_ylim()
ax1b.set_ylim(lo / (CPU_FREQ_GHZ * 1000.0), hi / (CPU_FREQ_GHZ * 1000.0))
ax1b.set_ylabel('FP64 за такт (FLOPS/cycle)', fontsize=11, color='#555')
ax1b.tick_params(axis='y', colors='#555')

snr_lo = df['SNR'].min()
snr_hi = df['SNR'].max()
pad = max((snr_hi - snr_lo) * 0.25, 0.3)
ax2.set_ylim(snr_lo - pad, snr_hi + pad)

fig.suptitle('Бенчмарк БПФ: производительность, эффективность и точность (FP64)',
             fontsize=14, y=0.965)

# 8. ЛЕГЕНДА (одна на обе панели, вынесена над графиком)
handles = [Line2D([], [], color=COLORS[f], linestyle='-', marker='o',
                  markerfacecolor=COLORS[f], linewidth=2.4, label=f) for f in FAMILIES]
handles += [
    Line2D([], [], color='0.35', linestyle='--', marker='o', markersize=6,
           markerfacecolor='white', markeredgewidth=1.4, linewidth=1.6,
           label='базовый (скалярный)'),
    Line2D([], [], color='0.35', linestyle='-', marker='o', markersize=7,
           markerfacecolor='0.35', linewidth=2.4, label='SIMD AVX2 + FMA (новый)'),
]
ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.015),
           ncol=6, shadow=True, framealpha=0.95, fontsize=9.5)

# 9. ИНФО-БЛОКИ
info_box = (f"{COMPILER} | {FLAGS}\n"
            f"CPU: i7-8565U @ {CPU_FREQ_GHZ} GHz (турбо-буст выключен)\n"
            f"Date: {datetime.now().strftime('%d.%m.%Y')}")
ax1.text(0.015, 0.98, info_box, transform=ax1.transAxes, fontsize=8, family='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

ax1.grid(True, which="major", axis='both', alpha=0.2, linestyle='--')
ax2.grid(True, which="major", axis='both', alpha=0.2, linestyle='--')

# 10. СОХРАНЕНИЕ
plt.savefig('fft_benchmark.png', dpi=300, bbox_inches='tight')
plt.show()
