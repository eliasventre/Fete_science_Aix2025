import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

"""
Simulazione modello TGI con dosaggio orale multiplo
PK a due compartimenti (orale)
PD log-kill con decadimento esponenziale della resistenza
Unità: giorni
TS = diametro del tumore [cm]
TS0 = 10 cm
Trattamento: 20 mg/die, 4 settimane on / 2 settimane off (dose_times generati come nel tuo script)
Simulazione totale: 365 giorni
Output: animazione 3 pannelli (sfera 3D + diametro + esposizione)
"""

# -----------------------------
# 1. PARAMETRI (unità in giorni)
# -----------------------------
ka = 3.024           # assorbimento orale [1/d]
CL = 818.4           # clearance centrale [L/d]
V1 = 2700            # volume compartimento centrale [L]
V2 = 774             # volume compartimento periferico [L]
Q = 16.512           # intercompartmental clearance [L/d]

kge = 0.001          # crescita tumorale esponenziale [1/d]
kkill = 0.1          # tasso killing farmaco [1/d]
lambda_res = 0.002   # decadimento efficacia (resistenza) [1/d]

TS0 = 1           # diametro iniziale del tumore [cm]

# -----------------------------
# 2. DOSE E REGIME TERAPEUTICO
# -----------------------------
dose = 100
treatment_duration = 252   # giorni totali di trattamento (come nel tuo script)
sim_duration = 365         # durata simulazione
# genera dose_times secondo schema 4w ON / 2w OFF (come nel tuo script)
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

# stato iniziale
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
    # somministra dose se siamo al tempo giusto (dose_times dati in giorni interi)
    # confronto con tolleranza perché t_current può non essere esattamente intero
    if dose_index < len(dose_times) and abs(t_current - dose_times[dose_index]) < 1e-6:
        y_current[0] += dose  # aggiunge dose in A_gut
        dose_index += 1

    # siamo in trattamento se t_current coincide con uno dei dose_times (usiamo isclose con atol)
    dose_active = np.any(np.isclose(t_current, dose_times, atol=0.5))

    # integra da t_current a t_next
    sol = solve_ivp(lambda tt, yy: tgi_model(tt, yy, dose_active=dose_active),
                    [t_current, t_next], y_current, t_eval=[t_next], method='RK45')

    # estrai stato finale
    y_current = sol.y[:, -1].tolist()
    t_current = t_next

    A_gut, Ac, Ap, TS, timeSinceTrtStart = y_current
    TS_list.append(TS)
    EXPOSURE_list.append(Ac / V1)
    t_list.append(t_current)

# arrays
t_array = np.array(t_list)
TS_array = np.array(TS_list)
EXPOSURE_array = np.array(EXPOSURE_list)

# -----------------------------
# 5. CREAZIONE ANIMAZIONE (SFERA 3D + GRAFICI DINAMICI)
# -----------------------------
fig = plt.figure(figsize=(14, 6))
# titolo globale (in francese come richiesto in precedenza)
fig.suptitle("Dose Forte - Fréquence Elevée - Durée Moyenne", fontsize=16, y=0.98)

# asse 3D per la sfera
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.set_xlim([-TS0, TS0])
ax3d.set_ylim([-TS0, TS0])
ax3d.set_zlim([-TS0, TS0])
ax3d.set_box_aspect([1,1,1])
ax3d.set_title("Évolution de la tumeur")

# mesh sfera
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
U, V = np.meshgrid(u, v)

# asse diametro (2D)
ax_diam = fig.add_subplot(2, 2, 2)
line_diam, = ax_diam.plot([], [], 'r', linewidth=2, label='Diamètre tumeur')  # diametro rosso
# linea diametro iniziale nera tratteggiata
ax_diam.axhline(TS0, color='k', linestyle='--', linewidth=1.5, label='Diamètre initial')
# linea verticale fine trattamento (azzurro chiaro)
ax_diam.axvline(treatment_duration, color='c', linestyle='--', linewidth=1.2, label='Fin du traitement')
ax_diam.set_xlim(0, sim_duration)
ax_diam.set_ylim(0, 1.5)
ax_diam.set_xlabel("Temps (jours)")
ax_diam.set_ylabel("Diamètre (cm)")
ax_diam.set_title("Diamètre au cours du temps")
ax_diam.set_box_aspect(1)
ax_diam.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# asse esposizione (2D)
ax_exp = fig.add_subplot(2, 2, 4)
line_exp, = ax_exp.plot([], [], 'g', linewidth=2, label='Exposition')
# linea tossicità massima blu
ax_exp.axhline(0.06, color='b', linestyle='--', linewidth=1.5, label='Toxicité maximale')
# regione tossica colorata (grigio chiaro)
# Nota: usiamo l'intero vettore t_array per disegnare la regione statica
ax_exp.fill_between(t_array, 0.06, 0.12, color='lightgrey', alpha=0.6, label='Zone toxique')
# linea verticale fine trattamento (azzurro chiaro)
ax_exp.axvline(treatment_duration, color='c', linestyle='--', linewidth=1.2, label='Fin du traitement')
ax_exp.set_xlim(0, sim_duration)
ax_exp.set_ylim(0, 0.12)
ax_exp.set_xlabel("Temps (jours)")
ax_exp.set_ylabel("Exposition")
ax_exp.set_title("Exposition au cours du temps")
ax_exp.set_box_aspect(1)
ax_exp.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# indici dei frame: mostriamo un frame ogni 10 giorni (accelera l'animazione)
frame_indices = np.arange(0, len(t_array), int(10/dt_step))
if frame_indices[-1] != len(t_array)-1:
    # assicuriamoci di includere l'ultimo frame
    frame_indices = np.concatenate([frame_indices, [len(t_array)-1]])

def update(frame_idx):
    frame = frame_indices[frame_idx]

    # aggiorna sfera 3D (rimuovendo precedenti surface)
    for coll in list(ax3d.collections):
        coll.remove()
    R = TS_array[frame] / 2.0
    X = R * np.cos(U) * np.sin(V)
    Y = R * np.sin(U) * np.sin(V)
    Z = R * np.cos(V)
    ax3d.plot_surface(X, Y, Z, color='red', alpha=0.6)
    ax3d.set_title(f"Tumeur - Jour {t_array[frame]:.1f}")

    # aggiorna linee 2D progressive
    # per evitare problemi di plotting su indici 0, usiamo frame+1
    line_diam.set_data(t_array[:frame+1], TS_array[:frame+1])
    line_exp.set_data(t_array[:frame+1], EXPOSURE_array[:frame+1])

    return ax3d, line_diam, line_exp

# ANIMATION SENZA RIPETIZIONE
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    interval=200,
    blit=False,
    repeat=False  # <-- non ripete l'animazione
)

plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


