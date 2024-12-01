import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.stats import gmean
from scipy.interpolate import CubicSpline

temp = float(input("Introduce la temperatura en grados centígrados: "))
opcion = int(input("Introduce 0 o 1, 0 si ajustas los datos en el script, 1 si va desde otro archivo: "))

# Función para cargar datos
def cargar_datos(opcion, similes=None):
    if opcion == 0:
        # Usar vectores definidos en el script
        t = np.array([0,44,85,127,177,227,275,323,372,426,485,536,602,664,729,800,872,940])
        h = np.array([0.245,0.235,0.225,0.215,0.205,0.195,0.185,0.175,0.165,0.155,0.145,0.135,0.125,0.115,0.105,0.095,0.085,0.075])
    elif opcion == 1:
        # Leer datos de un archivo proporcionado
        if similes.endswith('.txt') or similes.endswith('.dat'):
            datos = np.loadtxt(similes, delimiter=',')
        elif similes.endswith('.xlsx'):
            similes = pd.read_excel(similes).to_numpy()
        else:
            raise ValueError("Formato de archivo no soportado. Use .txt, .dat o .xlsx")
        t = datos[:, 0]
        h = datos[:, 1]
    else:
        raise ValueError("Opción no válida. Use 0 para vectores o 1 para archivo.")
    return t, h

# Cargar los datos
t, h = cargar_datos(opcion)

# Número de puntos por nodo (ventana para el spline)
n_puntos_por_nodo = 3

# Preparar listas para almacenar resultados
t_resultados = []
h_resultados = []
h_calculados_resultados = []
dh_dt_resultados = []
dh_dt2 = []
# Graficar los resultados
plt.figure(figsize=(8, 5))
plt.plot(t, h, 'o', label="Datos originales", color='black')

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
    cs2 = CubicSpline(t,h,bc_type='natural')
    dh_dt2_calc =(cs2.derivative()(t[i]))
    # Almacenar los resultados
    t_resultados.append(t[i])
    h_resultados.append(h[i])
    h_calculados_resultados.append(h_calculada)
    dh_dt_resultados.append(dh_dt)
    dh_dt2.append(dh_dt2_calc)
    # Crear el spline para la ventana actual
    cs = CubicSpline(t_nodos, h_nodos, bc_type='natural')
    t_fine = np.linspace(min(t_nodos), max(t_nodos), 100)  # Más puntos para suavidad
    h_fine = cs(t_fine)
    plt.plot(t_fine, h_fine, color='orange', alpha=0.5)

dh_dt_df = [f"{dh:.5f}" for dh in dh_dt_resultados]
# Convertir las listas en un DataFrame de pandas para mostrar los resultados de forma tabular
df_resultados = pd.DataFrame({
    't (min)': t_resultados,
    'h (original)': h_resultados,
    'h (calculada)': h_calculados_resultados,
    'dh/dt': dh_dt_df
})

display(df_resultados.to_string(index=False))
print("\n")
plt.xlabel('Tiempo (minutos)')
plt.ylabel('Altura (h)')
plt.title('Spline con nodos centrados y adyacentes')
plt.legend(["Datos originales", "Spline ajustado"])
plt.show()

# Análisis de logaritmos
log_h = np.log(h_calculados_resultados)
log_dh_dt = np.log(-np.array(dh_dt_resultados))  # Cambiar el signo para manejar el logaritmo
df_log = pd.DataFrame({
    'log(h)': log_h,
    'log(-dh/dt)': log_dh_dt
})
display(df_log.to_string(index=False))
print("\n")
# Ajuste lineal
slope, intercept, r_value, p_value, std_err = linregress(log_h, log_dh_dt)
ajuste = slope * log_h + intercept  # Línea ajustada
window_size = 3
rolling_mean = pd.Series(log_dh_dt).rolling(window=window_size,center=True).mean()

kexp = np.exp(intercept)

# Mostrar resultados del ajuste
print("Resultados del ajuste lineal:")
print(f"Pendiente: {slope:.3f}")
print(f"Intercepto: {intercept:.3f}", f"Valor de k: {kexp:.3f}")
print(f"Coeficiente de correlación (R²): {r_value**2:.3f}")
print("\n")

h_data = np.array(h_calculados_resultados)  # Altura calculada
dh_dt_data = -np.array(dh_dt_resultados)   # Derivada calculada (cambiada de signo)
def modelo(hc,k,n_reaccion):
    return k * hc**n_reaccion
parametros, covarianza = curve_fit(modelo, h_data, dh_dt_data, p0=[kexp, slope])  # p0: valores iniciales para k_n y n
k, n_reaccion = parametros
# ajustar k a multiplo de 0.5 
nround = round(n_reaccion/0.5) * 0.5
def error(param):
    k_opt, = param
    return np.sum((modelo(h_data, k_opt, nround) - dh_dt_data)**2)
resultado =minimize(error,x0=[k])
kopt = resultado.x
print(f"Valor ajustado para n multiplo de 0.5: {nround:.5f}, k ajustado: {kopt[0]:.5f}")
# Mostrar los resultados del ajuste
print(f"Valor ajustado de k_n: {k:.5f}")
print(f"Valor ajustado de n: {n_reaccion:.5f}")

# Calcular -dh/dt usando los parámetros ajustados
dh_dt_modelo = modelo(h_data, k, n_reaccion)
ss_res = np.sum((dh_dt_data - dh_dt_modelo)**2)  # Suma de los residuos cuadrados
ss_tot = np.sum((dh_dt_data - np.mean(dh_dt_data))**2)  # Suma de los cuadrados totales
r_squared = 1 - (ss_res / ss_tot)
dh_dt_ajuste = modelo(h_data, kexp, slope)
dh_dt_kopt = modelo(h_data, kopt, nround)
print(f"Coeficiente de determinación R²: {r_squared:.5f}")

# para spiline total
dh_dt2 = -np.array(dh_dt2)
parametros2, covarianza2 = curve_fit(modelo, h, dh_dt2, p0=[kexp, slope])
k2, n_reaccion2 = parametros2
print(f"el valor de n es: {n_reaccion2:.9f}")
nround2 = round(n_reaccion2/0.5) * 0.5
resultado2 =minimize(error,x0=[k2])
kopt2 = resultado2.x
print(f"Valor ajustado para n multiplo de 0.5: {nround2:.5f}, k ajustado spline total: {kopt2[0]:.5f}")
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# Subplot 1: Datos originales
axes[0].plot(log_h, log_dh_dt, 'o', label='Datos originales')
axes[0].plot(log_h, ajuste, '-', label=f'Ajuste: Pendiente={slope:.3f}, R²={r_value**2:.3f}', color='red')
axes[0].plot(log_h, rolling_mean, '-', label=f'Media móvil (ventana={window_size})', color='green', alpha=0.5)
axes[0].set_xlabel('log(h)')
axes[0].set_ylabel('log(-dh/dt)')
axes[0].set_title('Ajuste y media móvil para datos originales')
axes[0].legend()

# subplot 2: Datos con curve fit 
axes[1].scatter(h_data, dh_dt_data, label='Datos experimentales', color='blue')
axes[1].plot(h_data, dh_dt_modelo, label=f'Modelo ajustado: $k_n={k:.5f}, n={n_reaccion:.2f}$', color='red')
axes[1].plot(h_data, dh_dt_ajuste, label=f'Modelo inicial: $k_n={kexp:.5f}, n={slope:.2f}$', color='green')
axes[1].plot(h_data, dh_dt_kopt, label=f'Modelo ajustado multiplo de k: $k_n={kopt[0]:.5f}, n={nround:.2f}$', color='black')
axes[1].set_xlabel('h,m')
axes[1].set_ylabel('-dh/dt')
axes[1].legend()
axes[1].set_title('Ajuste de la ecuación $-dh/dt = k_n * h^n$')


plt.tight_layout()
plt.show()
if nround == 0:
    calc_h_opt = np.full_like(dh_dt_data, np.nan)
else:
    calc_h_opt = np.round((dh_dt_data/kopt)**(1/nround),3)
if nround2 != 0:
    calc_h_total = np.round((dh_dt2/kopt2)**(1/nround2),3)
elif nround != 0:
    calc_h_total = np.round((-dh_dt2/kopt)**(1/nround),3)
    kopt2, nround2 = kopt, nround
else:
    calc_h_total = np.full_like(dh_dt2, np.nan)
df_resultados['h_opt'] = calc_h_opt
df_resultados['h_sp_total'] = calc_h_total
df_resultados = df_resultados[df_resultados.columns[[0, 1, 2, -1, -2,-3]]]

display(df_resultados.to_string(index=False))
print("\n")
print("\n")
''' se comprueba el error cuadratico medio para ambos modelos
y se selecciona el que tenga menor error para compararlo con el 
metodo integral '''

valid_indices = ~np.isnan(h) & ~np.isnan(calc_h_opt) & ~np.isnan(calc_h_total)
h_valid = h[valid_indices]
calc_h_opt_valid = calc_h_opt[valid_indices]
calc_h_total_valid = calc_h_total[valid_indices]
mse_opt = np.mean((calc_h_opt_valid - h_valid) ** 2)
mse_total = np.mean((calc_h_total_valid - h_valid) ** 2)
if mse_opt < mse_total:
    h_derivada = calc_h_opt
    k_comparacion = kopt
    n_comparacion = nround
else:
    h_derivada = calc_h_total
    k_comparacion = kopt2
    n_comparacion = nround2

#! Método integral
epsilon = 1e-10
n_integral = np.array([0,0.5, 1, 1.5, 2])
x = np.zeros(len(h))
for i in range(1,len(h)):
    x[i]= (h[0]-h[i])/h[0]
x[0] = 0
ki_values = []
ki_value = np.zeros_like(t)
epsilon = 1e-10
for n in n_integral:
    if n == 1:
        ki_value = np.where((x != 0) & (t != 0),-np.log(np.clip(1 - x, epsilon, 1)) / np.clip(t, epsilon, None), 0)
    else:
        ki_value = np.where((x != 0) & (t != 0), (1 - (1 - x)**(1 - n)) / (1 - n) / np.clip(t, epsilon, None) / h[0]**(n - 1), 0)
    ki_values.append(ki_value)
ki_values = np.array(ki_values)

ki_transposed = ki_values.T
column_names = [f"n_reac {n}" for n in n_integral]
df = pd.DataFrame(ki_transposed, columns=column_names)
display(df.to_string(index=False))
print("\n")

#excluir primera fila para cada columna y asi calcular la dispersion
df_dispersion = df.iloc[1:,:].std()
min_dispersion = df_dispersion.idxmin()
column_values = df[min_dispersion].iloc[1:]


geom_mean = gmean(column_values)
err = abs(geom_mean - k_comparacion[0]) / np.maximum(abs(geom_mean), abs(k_comparacion[0]))
df_resultado_final = pd.DataFrame({
    'orden de reaccion metodo integral': min_dispersion, 
    'orden de reaccion metodo diferencial': n_comparacion,
    'k metodo integral': geom_mean,
    'k metodo diferencial': k_comparacion,
    'error relativo': "{:.2%}".format(err)
})

pd.set_option('display.colheader_justify', 'center')
display(df_resultado_final.to_string(index=False))

#calculo reynolds 
v = 1.19e-4-5.90e-6*temp+8.81e-8*temp**2

def calculo_dt(h):
    if n_comparacion == 0:
        dt = 0.073*h**0.5
    elif n_comparacion == 0.5:
        dt = 0.054*h**0.25
    elif n_comparacion == 1.5:
        dt = 0.0198*h**(-0.25)
    elif n_comparacion == 2:
        dt = 0.0098*h**(-0.5)
    return dt
dhdt2 = -k_comparacion[0] * h**n_comparacion
if n_comparacion != 0:
    hreynold = (dh_dt_data / k_comparacion[0]) ** (1 / n_comparacion)
    hreynold2 = (dhdt2 / k_comparacion[0]) ** (1 / n_comparacion)
else:
    hreynold = np.where((x != 0) & (t != 0),((1 - (1 - x)) / (k_comparacion[0] * t)) ** (-1),0)

reynolds =dh_dt_data*calculo_dt(h)/v
reynolds2 = -dhdt2*calculo_dt(h)/v
media_reynolds = gmean(reynolds[1:])
media_reynolds2 = gmean(reynolds2[1:])
print("\n")
print(f"la viscosidad para la temperatura {temp} es: {v:.3e} m^2/s")
print(f"El valor de reynolds para dhdt con spilines es: {media_reynolds:.5f}")
print(f"El valor de reynolds para dhdt = -k*h**n: {media_reynolds2:.5f}")
print("\n")
error_h = abs(h-hreynold)/np.maximum(h,hreynold)
error_h2 = abs(h-hreynold2)/np.maximum(h,hreynold2)

print(f"la h diff sera para el metodo diferencial salvo si el orden de reaccion es 0")
print(f"la h mixta usa el calculo de dhdt de la misma forma que para el segundo valor del reynolds")
print("\n")

def hintegral(h):
    if n_comparacion != 1:
        hintegral = np.where((t != 0), h[0] * ((1 - k_comparacion[0] * t * (1 - n_comparacion) * h[0]**(n_comparacion - 1))**(1 / (1 - n_comparacion))),h[0])
    else:
        hintegral = np.where((t != 0), ((np.exp(t * k_comparacion[0]) + 1) * h[0] - h[0]), h[0])
    return hintegral
hintegral_result = hintegral(h)
error_hintegral = abs(h-hintegral_result)/np.maximum(h,hintegral_result)
df_hfinal = pd.DataFrame({
    'h   ': h,
    'h diff   ': [f"{e:.3f}" for e in hreynold],
    'h integral   ': [f"{e:.3f}" for e in hintegral_result],
    'h mixta   ': [f"{e:.3f}" for e in hreynold2],
    'error relativo diff   ': [f"{e:.2%}" for e in error_h],
    'error relativo integral   ': [f"{e:.2%}" for e in error_hintegral],
    'error relativo mixta   ': [f"{e:.2e}%" for e in error_h2],
    'dt': [f"{e:.4f}" for e in calculo_dt(h)]
})
pd.set_option('display.colheader_justify', 'center')
display(df_hfinal.to_string(index=False))

# Graficar dh/dt
plt.figure(figsize=(8, 5))
plt.plot(t_resultados, dh_dt_resultados, marker='o', markersize=4, linestyle='-', color='blue', label="dh/dt calculado con spiline", linewidth=0.8)
plt.plot(t, -dh_dt2, marker='D', markersize=4, linestyle='-', color='red', label="dh/dt calculado con -k*h**n", linewidth=0.8,alpha=0.5)
plt.xlabel('Tiempo (minutos)')
plt.ylabel('dh/dt')
plt.title('Derivada de la altura con respecto al tiempo')
plt.legend()
plt.show()