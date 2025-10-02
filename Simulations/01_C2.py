import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1. PARAMÈTRES
# -----------------------------
ka = 3.024
CL = 818.4
V1 = 2700
V2 = 774
Q = 16.512

kge = 0.001
kkill = 0.1
lambda_res = 0.002
TS0 = 0.1 # diamètre initial [cm]

# -----------------------------
# 2. DOSES
# -----------------------------
dose = 20
duree_traitement = 252
intervalle_dose = 2.0
temps_doses = np.arange(0, duree_traitement + 0.1, intervalle_dose)
duree_simulation = 365
dt_step = 0.1  # pas d'intégration

# -----------------------------
# 3. MODÈLE ODE
# -----------------------------
def modele_tgi(t, y, dose_active=True):
    A_gut, Ac, Ap, TS, temps_depuis_traitement = y
    k12 = Q / V1
    k21 = Q / V2

    dA_gut = -ka * A_gut
    dAc = ka * A_gut - k12*Ac - (CL/V1)*Ac + k21*Ap
    dAp = k12*Ac - k21*Ap

    EXPOSURE = Ac / V1
    dtemps_depuis_traitement = 1.0 if dose_active else 0.0
    K = kkill * EXPOSURE * np.exp(-lambda_res * temps_depuis_traitement) * np.exp(-0.1*TS) if dose_active else 0.0

    if TS > 1e12:
        dTS = 0
    elif TS < 0.08:
        dTS = -K * TS
    else:
        dTS = kge * TS - K * TS

    return [dA_gut, dAc, dAp, dTS, dtemps_depuis_traitement]

# -----------------------------
# 4. SIMULATION
# -----------------------------
t_eval = np.arange(0, duree_simulation + dt_step, dt_step)

A_gut, Ac, Ap, TS, temps_depuis_traitement = 0.0, 0.0, 0.0, TS0, 0.0
TS_list, EXPOSURE_list, t_list = [], [], []

indice_dose = 0
for t in t_eval:
    if indice_dose < len(temps_doses) and abs(t - temps_doses[indice_dose]) < 1e-6:
        A_gut += dose
        indice_dose += 1

    dose_active = t <= duree_traitement

    y_current = [A_gut, Ac, Ap, TS, temps_depuis_traitement]
    sol = solve_ivp(lambda t, y: modele_tgi(t, y, dose_active=dose_active),
                    [0, dt_step], y_current, method='RK45', t_eval=[dt_step])
    A_gut, Ac, Ap, TS, temps_depuis_traitement = sol.y[:, -1]

    TS_list.append(TS)
    EXPOSURE_list.append(Ac / V1)
    t_list.append(t)

TS_array = np.array(TS_list)
EXPOSURE_array = np.array(EXPOSURE_list)
t_array = np.array(t_list)

# -----------------------------
# 5. ANIMATION: SPHÈRE + COURBES + RÉFÉRENCES + RÉGION TOXICITÉ + LIGNES FIN TRAITEMENT
# -----------------------------
fig = plt.figure(figsize=(14, 6))
fig.suptitle("Dose Faible - Fréquence Faible - Durée Moyenne", fontsize=16, y=0.98)

# 3D tumeur
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.set_xlim([-TS0, TS0])
ax3d.set_ylim([-TS0, TS0])
ax3d.set_zlim([-TS0, TS0])
ax3d.set_box_aspect([1,1,1])
ax3d.set_title("Évolution de la tumeur")

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
U, V = np.meshgrid(u, v)

# Subplot diamètre
ax_diam = fig.add_subplot(2, 2, 2)
line_diam, = ax_diam.plot([], [], 'r', label='Diamètre tumeur')
ax_diam.axhline(TS0, color='k', linestyle='--', label='Diamètre initial')
ax_diam.axvline(duree_traitement, color='c', linestyle='--', label='Fin du traitement')
ax_diam.set_xlim(0, duree_simulation)
ax_diam.set_ylim(0, 0.15)
ax_diam.set_xlabel("Temps (jours)")
ax_diam.set_ylabel("Diamètre (cm)")
ax_diam.set_title("Diamètre au cours du temps")
ax_diam.set_box_aspect(1)
ax_diam.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# Subplot exposition
ax_exp = fig.add_subplot(2, 2, 4)
line_exp, = ax_exp.plot([], [], 'g', label='Exposition')
ax_exp.axhline(0.06, color='b', linestyle='--', label='Toxicité maximale')
ax_exp.fill_between(t_array, 0.06, 0.12, color='lightgrey', alpha=0.5, label='Zone toxique')
ax_exp.axvline(duree_traitement, color='c', linestyle='--', label='Fin du traitement')
ax_exp.set_xlim(0, duree_simulation)
ax_exp.set_ylim(0, 0.12)
ax_exp.set_xlabel("Temps (jours)")
ax_exp.set_ylabel("Exposition")
ax_exp.set_title("Exposition au cours du temps")
ax_exp.set_box_aspect(1)
ax_exp.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# Frame tous les 10 jours
frame_indices = np.arange(0, len(t_array), int(10/dt_step))

def update(frame_idx):
    frame = frame_indices[frame_idx]

    # mise à jour sphère
    for coll in ax3d.collections:
        coll.remove()
    R = TS_array[frame]/2
    X = R * np.cos(U) * np.sin(V)
    Y = R * np.sin(U) * np.sin(V)
    Z = R * np.cos(V)
    ax3d.plot_surface(X, Y, Z, color='red', alpha=0.6)
    ax3d.set_title(f"Tumeur - Jour {t_array[frame]:.1f}")

    # mise à jour courbes
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