import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DE LA SEÑAL EN EL TIEMPO
# ==========================================
fs = 1000            # Frecuencia de muestreo (Hz)
T = 1.0              # Duración total (segundos)
t = np.arange(0, T, 1/fs)

# Creamos una señal compuesta por dos senoidales (ej: 50 Hz y 120 Hz)
f1, f2 = 50, 120
señal_tiempo = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# ==========================================
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER
# ==========================================
N = len(señal_tiempo)
# Aplicamos la Transformada Rápida de Fourier (FFT)
fft_resultado = np.fft.fft(señal_tiempo)
# Calculamos el espectro de magnitudes normalizado
fft_mag = (2/N) * np.abs(fft_resultado[:N // 2])
# Vector de frecuencias correspondientes
frecuencias = np.fft.fftfreq(N, 1/fs)[:N // 2]

# ==========================================
# 3. VISUALIZACIÓN DE LOS RESULTADOS
# ==========================================
plt.figure(figsize=(12, 8))

# Gráfica en el dominio del tiempo
plt.subplot(2, 1, 1)
plt.plot(t[:200], señal_tiempo[:200], color='tab:blue')
plt.title("1. Representación de la señal en el Dominio del Tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# Gráfica en el dominio de la frecuencia (Espectro)
plt.subplot(2, 1, 2)
plt.plot(frecuencias, fft_mag, color='tab:orange')
plt.title("2. Espectro de Frecuencia (Transformada de Fourier)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.xlim(0, 200) # Acotamos para ver claramente los picos en 50 Hz y 120 Hz

plt.tight_layout()
plt.show()