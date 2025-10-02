"""
Simulazione modello TGI con dosaggio orale multiplo
PK a due compartimenti (orale)
PD log-kill con decadimento esponenziale della resistenza
Unità: giorni
TS = diametro del tumore [cm]
TS0 = 10 cm
Trattamento: 20 mg/die, 4 settimane on / 2 settimane off, solo primi 42 giorni
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

kge = 0.001     # crescita tumorale esponenziale [1/d]
kkill = 0.1   # tasso killing farmaco [1/d]
lambda_res = 0.002 # decadimento efficacia (resistenza) [1/d]

TS0 = 10.0           # diametro iniziale del tumore [cm]

# -----------------------------
# 2. DOSE E REGIME TERAPEUTICO
# -----------------------------
dose = 100
treatment_duration = 84   # giorni totali di trattamento
sim_duration = 365        # giorni totali

# genera dose_times secondo schema 4w ON / 2w OFF (solo fino a 42 giorni)
dose_times = []
t = 0
while t < treatment_duration:
    # 4 settimane ON (28 giorni)
    for d in range(28):
        if t + d < treatment_duration:
            dose_times.append(t + d)
    t += 42  # salta 2 settimane OFF
dose_times = np.array(dose_times)

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
    dtimeSinceTrtStart = 1.0 if dose_active else 0.0

    K = kkill * EXPOSURE * np.exp(-lambda_res * timeSinceTrtStart) * np.exp(-0.1*TS) if dose_active else 0.0

    if TS > 1e12:
        dTS = 0
    elif TS < 0.08:
        dTS = -K * TS
    else:
        dTS = kge * TS - K * TS

    return [dA_gut, dAc, dAp, dTS, dtimeSinceTrtStart]

# -----------------------------
# 4. SIMULAZIONE STEP PER STEP
# -----------------------------
dt_step = 0.1
t_eval = np.arange(0, sim_duration + dt_step, dt_step)

A_gut = 0.0
Ac = 0.0
Ap = 0.0
TS = TS0
timeSinceTrtStart = 0.0
y_current = [A_gut, Ac, Ap, TS, timeSinceTrtStart]

TS_list = []
EXPOSURE_list = []
t_list = []

t_current = 0.0
dose_index = 0

for t_next in t_eval[1:]:
    # somministra dose se siamo al tempo giusto
    if dose_index < len(dose_times) and abs(t_current - dose_times[dose_index]) < 1e-6:
        y_current[0] += dose  # aggiunge dose in A_gut
        dose_index += 1

    # siamo in trattamento se t_current è tra i dose_times
    dose_active = np.any(np.isclose(t_current, dose_times, atol=0.5))

    # integra da t_current a t_next
    sol = solve_ivp(lambda tt, yy: tgi_model(tt, yy, dose_active=dose_active),
                    [t_current, t_next], y_current, t_eval=[t_next], method='RK45')

    # estrai stato finale (compatibile con array/lista)
    if hasattr(sol.y, "shape"):
        y_current = sol.y[:, -1].tolist()
    else:
        y_current = [yy[-1] for yy in sol.y]

    t_current = t_next

    # salva risultati
    A_gut, Ac, Ap, TS, timeSinceTrtStart = y_current
    TS_list.append(TS)
    EXPOSURE_list.append(Ac / V1)
    t_list.append(t_current)

# -----------------------------
# 5. GRAFICI
# -----------------------------
t_array = np.array(t_list)
TS_array = np.array(TS_list)
EXPOSURE_array = np.array(EXPOSURE_list)

plt.figure(figsize=(10,6))
plt.plot(t_array, TS_array, label="Diamètre de la tumeur (TS) [cm]")
plt.axvline(x=treatment_duration, color='red', linestyle='--', label="Fin du traitement")
plt.xlabel("Temps [jours]")
plt.ylabel("Diamètre de la tumeur [cm]")
plt.title("Dose Forte - Fréquence Elevée - Durée Courte")
plt.legend()
plt.grid(True)
plt.show(block=False)

plt.figure(figsize=(10,6))
plt.plot(t_array, EXPOSURE_array, label="Exposition", color='green')
plt.axvline(x=treatment_duration, color='red', linestyle='--', label="Fin du traitement")
plt.xlabel("Temps [jours]")
plt.ylabel("Exposition au médicament")
plt.ylim([0, 0.12])
plt.title("Dose Forte - Fréquence Elevée - Durée Courte")
plt.legend()
plt.grid(True)
plt.show()


