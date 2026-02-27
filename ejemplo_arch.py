import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# ==========================
# 1. SIMULAR DATOS FINANCIEROS
# ==========================
np.random.seed(42)

n = 1000
errores = np.random.normal(0, 1, n)

volatilidad_real = np.zeros(n)
retornos = np.zeros(n)

volatilidad_real[0] = 1

for t in range(1, n):
    volatilidad_real[t] = 0.1 + 0.3 * errores[t-1]**2 + 0.6 * volatilidad_real[t-1]
    retornos[t] = errores[t] * np.sqrt(volatilidad_real[t])

# ==========================
# 2. GRAFICAR RETORNOS
# ==========================
plt.figure(figsize=(10,4))
plt.plot(retornos)
plt.title("Serie de retornos simulados")
plt.xlabel("Tiempo")
plt.ylabel("Retorno")
plt.show()

# ==========================
# 3. AJUSTAR MODELO GARCH(1,1)
# ==========================
modelo = arch_model(retornos, vol='Garch', p=1, q=1)
resultado = modelo.fit()

print(resultado.summary())

# ==========================
# 4. VOLATILIDAD ESTIMADA
# ==========================
volatilidad_estimada = resultado.conditional_volatility

plt.figure(figsize=(10,4))
plt.plot(volatilidad_real, label="Volatilidad real")
plt.plot(volatilidad_estimada, label="Volatilidad estimada")
plt.title("Volatilidad: real vs estimada")
plt.legend()
plt.show()

# ==========================
# 5. PRONÓSTICO DE VOLATILIDAD
# ==========================
forecast = resultado.forecast(horizon=10)

print("\nPronóstico de varianza futura:")
print(forecast.variance[-1:])