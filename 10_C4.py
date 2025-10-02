"""
Simulazione modello TGI con dosaggio orale multiplo semplice
PK a due compartimenti (orale)
PD log-kill con decadimento esponenziale della resistenza
Unità: giorni
TS = diametro del tumore [cm]
TS0 = 10 cm
Trattamento: 20 mg ogni 2 giorni per i primi 42 giorni
Simulazione totale: 365 giorni
Grafici: Tumor Diameter (TS) e Exposure (Cc)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETRI (unità in giorni)
# -----------------------------
ka = 3.024           # assorbimento orale [1/d]
CL = 818.4           # clearance centrale [L/d]
V1 = 2700            # volume compartimento centrale [L]
V2 = 774             # volume compartimento periferico [L]
Q = 16.512           # intercompartmental clearance [L/d]

kge = 5.34e-04       # crescita tumorale esponenziale [1/d]
kkill = 3.46e-03     # tasso killing farmaco [1/d]
lambda_res = 1.32e-02 # decadimento efficacia (resistenza) [1/d]

TS0 = 10          # diametro iniziale del tumore [cm]

# -----------------------------
# 2. DOSE E REGIME TERAPEUTICO
# -----------------------------
dose = 50
treatment_duration = 42
dose_interval = 2.0
dose_times = np.arange(0, treatment_duration+0.1, dose_interval) # tempi delle dosi
sim_duration = 365

# -----------------------------
# 3. FUNZIONE ODE
# -----------------------------
def tgi_model(t, y, dose_active=True):
    A_gut, Ac, Ap, TS, timeSinceTrtStart = y
    k12 = Q / V1
    k21 = Q / V2

    dA_gut = -ka * A_gut
    dAc = ka * A_gut - k12*Ac - (CL/V1)*Ac + k21*Ap
    dAp = k12*Ac - k21*Ap

    EXPOSURE = Ac / V1

    # la variabile viene integrata da solve_ivp
    dtimeSinceTrtStart = 1.0 if dose_active else 0.0

    # uso direttamente timeSinceTrtStart (non "new")
    K = kkill * EXPOSURE * np.exp(-lambda_res * timeSinceTrtStart) * np.exp(-0.1*TS) if dose_active else 0.0

    if TS > 1e12:
        dTS = 0
    elif TS < 0.08:
        dTS = -K * TS
    else:
        dTS = kge * TS - K * TS

    return [dA_gut, dAc, dAp, dTS, dtimeSinceTrtStart]

# -----------------------------
# 4. SIMULAZIONE STEP PER STEP CON DOSI
# -----------------------------
dt_step = 0.1  # passo di integrazione [giorni]
t_eval = np.arange(0, sim_duration+dt_step, dt_step)

# inizializza
A_gut = 0.0
Ac = 0.0
Ap = 0.0
TS = TS0
timeSinceTrtStart = 0.0

TS_list = []
EXPOSURE_list = []
t_list = []

dose_index = 0

for t in t_eval:
    # somministra dose se siamo al tempo giusto
    if dose_index < len(dose_times) and abs(t - dose_times[dose_index]) < 1e-6:
        A_gut += dose
        dose_index += 1

    # determina se siamo nel periodo di trattamento
    dose_active = t <= treatment_duration

    # integra un piccolo step dt_step
    y_current = [A_gut, Ac, Ap, TS, timeSinceTrtStart]
    sol = solve_ivp(lambda t, y: tgi_model(t, y, dose_active=dose_active),
                    [0, dt_step], y_current, method='RK45', t_eval=[dt_step])
    A_gut, Ac, Ap, TS, timeSinceTrtStart = sol.y[:, -1]

    # salva risultati
    TS_list.append(TS)
    EXPOSURE_list.append(Ac / V1)
    t_list.append(t)

# converti in array
t_array = np.array(t_list)
TS_array = np.array(TS_list)
EXPOSURE_array = np.array(EXPOSURE_list)

# -----------------------------
# 5. GRAFICI
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(t_array, TS_array, label="Tumor Diameter (TS) [cm]")
plt.axvline(x=treatment_duration, color='red', linestyle='--', label="Fine trattamento")
plt.xlabel("Time [days]")
plt.ylabel("Tumor Diameter [cm]")
plt.title("TGI Model: Tumor Diameter over 365 days (42 giorni trattamento)")
plt.legend()
plt.grid(True)
plt.show(block=False)

plt.figure(figsize=(10,6))
plt.plot(t_array, EXPOSURE_array, label="Exposure (Cc) [mg/L]", color='green')
plt.axvline(x=treatment_duration, color='red', linestyle='--', label="Fine trattamento")
plt.xlabel("Time [days]")
plt.ylabel("Drug Exposure [mg/L]")
plt.ylim([0, 0.12])
plt.title("TGI Model: Drug Exposure over 365 days (42 giorni trattamento)")
plt.legend()
plt.grid(True)
plt.show()
