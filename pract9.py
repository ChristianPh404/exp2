import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.interpolate import CubicSpline
# Vectores t y h
t = np.array([
    0,44,85,127,177,227,275,323,372,426,485,536,602,664,729,800,872,940
])

h = np.array([
    0.245,0.235,0.225,0.215,0.205,0.195,0.185,0.175,0.165,0.155,0.145,0.135,0.125,0.115,0.105,0.095,0.085,0.075
])

# Número de puntos por nodo (ventana para el spline)
n_puntos_por_nodo = 3

# Preparar listas para almacenar resultados
t_resultados = []
h_resultados = []
h_calculados_resultados = []
dh_dt_resultados = []

# Para cada punto de tiempo
for i in range(len(t)):
    # Definir ventana de nodos alrededor del punto actual
    if i < n_puntos_por_nodo // 2:  # Si estamos al inicio
        t_nodos = t[:n_puntos_por_nodo]
        h_nodos = h[:n_puntos_por_nodo]
        indice = i  # Índice correspondiente al punto actual
    elif i >= len(t) - n_puntos_por_nodo // 2:  # Si estamos al final
        t_nodos = t[-n_puntos_por_nodo:]
        h_nodos = h[-n_puntos_por_nodo:]
        indice = i - (len(t) - n_puntos_por_nodo)  # Índice relativo en la ventana
    else:  # Si estamos en el centro
        t_nodos = t[i - n_puntos_por_nodo // 2: i + n_puntos_por_nodo // 2 + 1]
        h_nodos = h[i - n_puntos_por_nodo // 2: i + n_puntos_por_nodo // 2 + 1]
        indice = n_puntos_por_nodo // 2  # Índice central

    # Crear el spline cúbico para la ventana actual
    cs = CubicSpline(t_nodos, h_nodos, bc_type='natural')
    
    # Calcular la derivada y el valor interpolado en el punto actual
    dh_dt = cs.derivative()(t[i])  # Derivada en el punto actual
    h_calculada = cs(t[i])  # Evaluar el spline en el punto actual
    
    # Almacenar los resultados
    t_resultados.append(t[i])
    h_resultados.append(h[i])
    h_calculados_resultados.append(h_calculada)
    dh_dt_resultados.append(dh_dt)

dh_dt_df = [f"{dh:.5f}" for dh in dh_dt_resultados]
# Convertir las listas en un DataFrame de pandas para mostrar los resultados de forma tabular
df_resultados = pd.DataFrame({
    't (min)': t_resultados,
    'h (original)': h_resultados,
    'h (calculada)': h_calculados_resultados,
    'dh/dt': dh_dt_df
})
# Mostrar el DataFrame
display(df_resultados)

# Graficar los resultados
plt.figure(figsize=(8, 5))
plt.plot(t, h, 'o', label="Datos originales", color='black')

# Graficar el spline ajustado para cada conjunto de nodos
n_puntos_por_nodo = 3
for i in range(len(t)):
    # Determinar los nodos relevantes
    if i < n_puntos_por_nodo // 2:  # Al inicio
        t_nodos = t[:n_puntos_por_nodo]
        h_nodos = h[:n_puntos_por_nodo]
    elif i >= len(t) - n_puntos_por_nodo // 2:  # Al final
        t_nodos = t[-n_puntos_por_nodo:]
        h_nodos = h[-n_puntos_por_nodo:]
    else:  # Centro
        t_nodos = t[i - n_puntos_por_nodo // 2: i + n_puntos_por_nodo // 2 + 1]
        h_nodos = h[i - n_puntos_por_nodo // 2: i + n_puntos_por_nodo // 2 + 1]

    # Crear el spline para la ventana actual
    cs = CubicSpline(t_nodos, h_nodos, bc_type='natural')
    t_fine = np.linspace(min(t_nodos), max(t_nodos), 100)  # Más puntos para suavidad
    h_fine = cs(t_fine)
    plt.plot(t_fine, h_fine, color='orange', alpha=0.5)

plt.xlabel('Tiempo (minutos)')
plt.ylabel('Altura (h)')
plt.title('Spline con nodos centrados y adyacentes')
plt.legend(["Datos originales", "Spline ajustado"])
plt.show()

# Graficar dh/dt
plt.figure(figsize=(8, 5))
plt.plot(t_resultados, dh_dt_resultados, 'o-', label="dh/dt calculado", color='blue')
plt.xlabel('Tiempo (minutos)')
plt.ylabel('dh/dt')
plt.title('Derivada de la altura con respecto al tiempo')
plt.legend()
plt.show()

# Análisis de logaritmos
log_h = np.log(h_calculados_resultados)
log_dh_dt = np.log(-np.array(dh_dt_resultados))  # Cambiar el signo para manejar el logaritmo
df_log = pd.DataFrame({
    'log(h)': log_h,
    'log(-dh/dt)': log_dh_dt
})

plt.figure(figsize=(8, 5))
plt.plot(log_h, log_dh_dt, 'o', label='Datos')

# Ajuste lineal
slope, intercept, r_value, p_value, std_err = linregress(log_h, log_dh_dt)
ajuste = slope * log_h + intercept  # Línea ajustada
window_size = 3
rolling_mean = pd.Series(log_dh_dt).rolling(window=window_size,center=True).mean()


plt.plot(log_h, rolling_mean, '-', label=f'Media móvil (ventana={window_size})', color='green', alpha=0.5)
plt.plot(log_h, ajuste, '-', label=f'Ajuste: Pendiente={slope:.3f}, R²={r_value**2:.3f}', color='red')


plt.xlabel('log(h)')
plt.ylabel('log(-dh/dt)')
plt.title('Ajuste lineal en espacio log-log')
plt.legend()
plt.show()
kexp = np.exp(intercept)


# Mostrar resultados del ajuste
print("Resultados del ajuste lineal:")
print(f"Pendiente: {slope:.3f}")
print(f"Intercepto: {intercept:.3f}", f"Valor de k: {kexp:.3f}")

print(f"Coeficiente de correlación (R²): {r_value**2:.3f}")

h_data = np.array(h_calculados_resultados)  # Altura calculada
dh_dt_data = -np.array(dh_dt_resultados)   # Derivada calculada (cambiada de signo)
def modelo(hc,k,n_reaccion):
    return k * hc**n_reaccion
parametros, covarianza = curve_fit(modelo, h_data, dh_dt_data, p0=[kexp, slope])  # p0: valores iniciales para k_n y n
k, n_reaccion = parametros
# Mostrar los resultados del ajuste
print(f"Valor ajustado de k_n: {k:.5f}")
print(f"Valor ajustado de n: {n_reaccion:.5f}")

# Calcular -dh/dt usando los parámetros ajustados
dh_dt_modelo = modelo(h_data, k, n_reaccion)
ss_res = np.sum((dh_dt_data - dh_dt_modelo)**2)  # Suma de los residuos cuadrados
ss_tot = np.sum((dh_dt_data - np.mean(dh_dt_data))**2)  # Suma de los cuadrados totales
r_squared = 1 - (ss_res / ss_tot)
dh_dt_ajuste = modelo(h_data, kexp, slope)
print(f"Coeficiente de determinación R²: {r_squared:.5f}")
plt.figure(figsize=(8, 5))
plt.scatter(h_data, dh_dt_data, label='Datos experimentales', color='blue')
plt.plot(h_data, dh_dt_modelo, label=f'Modelo ajustado: $k_n={k:.5f}, n={n_reaccion:.2f}$', color='red')
plt.plot(h_data, dh_dt_ajuste, label=f'Modelo inicial: $k_n={kexp:.5f}, n={slope:.2f}$', color='green')
plt.xlabel('h')
plt.ylabel('-dh/dt')
plt.legend()
plt.title('Ajuste de la ecuación $-dh/dt = k_n * h^n$')
plt.show()