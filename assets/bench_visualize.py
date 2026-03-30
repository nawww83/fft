import matplotlib.pyplot as plt

# Данные замеров
n_values = [1024, 4096, 16384, 65536]
n_labels = ['1k', '4k', '16k', '64k']
iter_mflops = [6545.8, 5519.1, 4655.7, 3338.2]
recur_mflops = [4996.3, 4693.3, 4581.8, 3153.7]

fig, ax1 = plt.subplots(figsize=(11, 7), dpi=120)

# Основные графики
ax1.plot(n_values, iter_mflops, 'o-', label='Iterative FFT', 
         linewidth=3, markersize=9, color='#1f77b4', alpha=0.9)
ax1.plot(n_values, recur_mflops, 's--', label='Recursive FFT (Hybrid)', 
         linewidth=3, markersize=9, color='#ff7f0e', alpha=0.9)

# Настройка осей
ax1.set_xscale('log', base=2)
ax1.set_xticks(n_values)
ax1.set_xticklabels(n_labels, fontsize=12)
ax1.set_xlabel('Vector Size (N)', fontsize=13, fontweight='bold', labelpad=10)
ax1.set_ylabel('Performance (MFLOPS)', fontsize=13, fontweight='bold', labelpad=10)

# Динамический предел Y для чистоты подписей
ax1.set_ylim(2500, 7500)

# Подписи Jitter (крупно и четко)
ax1.text(1024, 6900, "Jitter: 29.7%", fontsize=11, ha='center', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#1f77b4', boxstyle='round,pad=0.5'))

ax1.text(65536, 3550, "Jitter: 4.8%", fontsize=11, ha='center', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ff7f0e', boxstyle='round,pad=0.5'))

# Подпись про Base Case (указываем на гибридность)
ax1.annotate('Hybrid Architecture\n(Base Case N=32)', 
             xy=(4096, 4693.3), xytext=(2000, 3500),
             fontsize=11, color='#d35400', fontweight='bold',
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='#d35400', lw=2))

# Оформление заголовка и сетки
ax1.set_title('FFT Performance: Iterative vs Recursive\nIntel Core i7-8565U (Whiskey Lake)', 
             fontsize=15, pad=25, fontweight='bold')
ax1.grid(True, which="major", ls="--", alpha=0.5)
ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=12)

# Дополнительная шкала GFLOPS справа
ax2 = ax1.twinx()
ax2.set_ylabel('GFLOPS', fontsize=12, alpha=0.6, fontweight='bold')
ax2.set_ylim(ax1.get_ylim()[0]/1000, ax1.get_ylim()[1]/1000)

plt.tight_layout()
plt.savefig('fft_benchmark_final.png')
print("График fft_benchmark_final.png успешно создан.")
plt.show()
