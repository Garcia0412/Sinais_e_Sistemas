# Exercício Extra – Blue Whale
# ALUNO: VINÍCIUS LIMA GARCIA - 12421EEL009
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import fft
import sounddevice as sd
from scipy.fft import rfftfreq

# Path
BASE_DIR = Path(__file__).resolve().parent

# Leitura do arquivo .au
whaleFile = BASE_DIR / "bluewhale.au.au"
X, Fs = sf.read(whaleFile)

# Definindo a pasta de destino para salvar os gráficos
pasta_destino = 'Resultados'
if not os.path.exists(pasta_destino): # Confere se a pasta não existe
    os.makedirs(pasta_destino) # Cria a pasta se não existir

# Converter para mono caso seja estéreo
if len(X.shape) > 1:
    X = X[:, 0]

# Reproduzir áudio
sd.play(X, Fs)
sd.wait()


# Figura 1 – Sinal completo
# ======================
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(X)
plt.xlabel('Número de amostras')
plt.ylabel('Amplitude')
plt.title('Sinal Completo')


# Separando trecho (primeiro "moan")
# ======================
whale = X[int(2.45e4):int(3.10e4)]

# Vetor de tempo (corrigindo aceleração no tempo)
t = 10 * np.arange(len(whale)) / Fs

plt.subplot(2, 1, 2)
plt.plot(t, whale)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Trecho Selecionado')
plt.tight_layout()
plt.savefig("Resultados/Som_original.png", dpi=300, bbox_inches='tight')
plt.show()


# Transformada de Fourier
# ======================
N = len(whale)
Fs = Fs / 10          # corrigindo a frequência de amostragem após aceleração temporal
Y = fft(whale)
P1 = np.abs(Y/N)  # Normalizando a amplitude
P2 = P1[:N//2 + 1]  # Pegando apenas a metade positiva do espectro
P2[1:-1] = 2*P2[1:-1]  # Corrigindo a amplitude para as frequências positivas

f = rfftfreq(N, 1/Fs)  # Frequências correspondentes

plt.figure(2)
plt.plot(f, P2)
plt.title("Espectro do Sinal - Unilateral")
plt.xlabel("f (Hz)")
plt.ylabel("|X(f)|")
plt.xlim([0, 50])                   # Zoom, espectro vai até 500 Hz
plt.grid(True)
plt.savefig("Resultados/Espectro_Unilateral.png", dpi=300, bbox_inches='tight')
plt.show()