import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import dual_annealing
import warnings
import datetime
import numba
import pandas as pd
import os
import io
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)
@numba.njit(cache=True, fastmath=True)
def _calculate_polarization_numba(pol_type_is_s, nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, alpha, cos_theta, cos_theta_safe_p, num_layers):
    indices = np.empty(num_layers, dtype=np.complex128)
    for i in range(num_layers):
        if i % 2 == 0:
            indices[i] = nH_complex
        else:
            indices[i] = nL_complex
    N_angles = len(angles_rad)
    M = np.empty((N_angles, 2, 2), dtype=np.complex128)
    for k in range(N_angles):
        M[k, 0, 0] = 1.0 + 0.0j
        M[k, 0, 1] = 0.0 + 0.0j
        M[k, 1, 0] = 0.0 + 0.0j
        M[k, 1, 1] = 1.0 + 0.0j
    valid = np.ones(N_angles, dtype=np.bool_)
    term_sqrt_safe = np.empty(N_angles, dtype=np.complex128)
    eta_safe = np.empty(N_angles, dtype=np.complex128)
    Mi = np.empty((N_angles, 2, 2), dtype=np.complex128)
    for i in range(num_layers):
        Ni = indices[i]
        epi = epaisseur
        term_sqrt_sq = Ni**2 - alpha**2
        term_sqrt = np.sqrt(term_sqrt_sq)
        for k in range(N_angles):
            if ~np.isfinite(term_sqrt[k].real) or ~np.isfinite(term_sqrt[k].imag) or (np.abs(term_sqrt[k].real) < 1e-15 and np.abs(term_sqrt[k].imag) < 1e-15):
                valid[k] = False
                term_sqrt_safe[k] = 1.0 + 0j
            else:
                term_sqrt_safe[k] = term_sqrt[k]
        if pol_type_is_s:
            eta = term_sqrt_safe
        else:
            if np.abs(Ni.real) < 1e-15 and np.abs(Ni.imag) < 1e-15:
                valid[:] = False
                eta = np.zeros(N_angles, dtype=np.complex128)
            else:
                eta_denom_safe = np.empty_like(term_sqrt_safe)
                for k in range(N_angles):
                     if valid[k] and not (np.abs(term_sqrt_safe[k].real) < 1e-15 and np.abs(term_sqrt_safe[k].imag) < 1e-15):
                         eta_denom_safe[k] = term_sqrt_safe[k]
                     elif valid[k]:
                         valid[k] = False
                         eta_denom_safe[k] = 1.0 + 0j
                     else:
                         eta_denom_safe[k] = 1.0 + 0j
                inv_eta_denom_safe = np.zeros_like(eta_denom_safe)
                for k in range(N_angles):
                    if valid[k] and not (np.abs(eta_denom_safe[k].real) < 1e-15 and np.abs(eta_denom_safe[k].imag) < 1e-15):
                        inv_eta_denom_safe[k] = 1.0 / eta_denom_safe[k]
                    elif valid[k]:
                        valid[k] = False
                eta = (Ni**2) * inv_eta_denom_safe
        for k in range(N_angles):
             if not valid[k] or ~np.isfinite(eta[k].real) or ~np.isfinite(eta[k].imag) or (np.abs(eta[k].real) < 1e-15 and np.abs(eta[k].imag) < 1e-15):
                 valid[k] = False
                 eta_safe[k] = 1.0 + 0j
             else:
                 eta_safe[k] = eta[k]
        phi = (2 * np.pi / l0) * term_sqrt_safe * epi
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        inv_eta_safe = np.zeros_like(eta_safe)
        for k in range(N_angles):
            if valid[k] and not (np.abs(eta_safe[k].real) < 1e-15 and np.abs(eta_safe[k].imag) < 1e-15):
                inv_eta_safe[k] = 1.0 / eta_safe[k]
            elif valid[k]:
                valid[k] = False
        Mi[:, 0, 0] = cos_phi
        Mi[:, 0, 1] = 1j * inv_eta_safe * sin_phi
        Mi[:, 1, 0] = 1j * eta_safe * sin_phi
        Mi[:, 1, 1] = cos_phi
        M_prev = M.copy()
        M_new = np.empty_like(M)
        for k in range(N_angles):
            if not valid[k]:
                M_new[k] = M_prev[k]
                continue
            mi_is_finite = (np.isfinite(Mi[k,0,0].real) and np.isfinite(Mi[k,0,0].imag) and
                              np.isfinite(Mi[k,0,1].real) and np.isfinite(Mi[k,0,1].imag) and
                              np.isfinite(Mi[k,1,0].real) and np.isfinite(Mi[k,1,0].imag) and
                              np.isfinite(Mi[k,1,1].real) and np.isfinite(Mi[k,1,1].imag))
            if not mi_is_finite:
                valid[k] = False
                M_new[k] = M_prev[k]
                continue
            a, b = Mi[k, 0, 0], Mi[k, 0, 1]
            c, d = Mi[k, 1, 0], Mi[k, 1, 1]
            e, f = M_prev[k, 0, 0], M_prev[k, 0, 1]
            g, h = M_prev[k, 1, 0], M_prev[k, 1, 1]
            m00 = a * e + b * g
            m01 = a * f + b * h
            m10 = c * e + d * g
            m11 = c * f + d * h
            M_new[k, 0, 0] = m00
            M_new[k, 0, 1] = m01
            M_new[k, 1, 0] = m10
            M_new[k, 1, 1] = m11
            m_new_is_finite = (np.isfinite(m00.real) and np.isfinite(m00.imag) and
                               np.isfinite(m01.real) and np.isfinite(m01.imag) and
                               np.isfinite(m10.real) and np.isfinite(m10.imag) and
                               np.isfinite(m11.real) and np.isfinite(m11.imag))
            if not m_new_is_finite:
                valid[k] = False
                M_new[k] = M_prev[k]
        M = M_new
    eta_sub_term_sq = nSub_complex**2 - alpha**2
    etasub_term = np.sqrt(eta_sub_term_sq)
    etainc = np.empty(N_angles, dtype=np.complex128)
    etasub = np.empty(N_angles, dtype=np.complex128)
    if pol_type_is_s:
        etainc = cos_theta
        for k in range(N_angles):
            if not valid[k] or ~np.isfinite(etasub_term[k].real) or ~np.isfinite(etasub_term[k].imag):
                valid[k] = False
                etasub[k] = 1.0 + 0j
            else:
                 etasub[k] = etasub_term[k]
    else:
        inv_cos_theta_safe_p = np.zeros_like(cos_theta_safe_p)
        for k in range(N_angles):
            if valid[k] and not (np.abs(cos_theta_safe_p[k].real) < 1e-15 and np.abs(cos_theta_safe_p[k].imag) < 1e-15):
                 inv_cos_theta_safe_p[k] = 1.0 / cos_theta_safe_p[k]
            elif valid[k]:
                 valid[k] = False
        etainc = inv_cos_theta_safe_p
        if np.abs(nSub_complex.real) < 1e-15 and np.abs(nSub_complex.imag) < 1e-15:
             valid[:] = False
             etasub[:] = 0.0 + 0j
        else:
             etasub_term_safe = np.empty_like(etasub_term)
             for k in range(N_angles):
                 if not valid[k] or ~np.isfinite(etasub_term[k].real) or ~np.isfinite(etasub_term[k].imag) or (np.abs(etasub_term[k].real) < 1e-15 and np.abs(etasub_term[k].imag) < 1e-15):
                     valid[k] = False
                     etasub_term_safe[k] = 1.0 + 0j
                 else:
                     etasub_term_safe[k] = etasub_term[k]
             inv_etasub_term_safe = np.zeros_like(etasub_term_safe)
             for k in range(N_angles):
                 if valid[k] and not (np.abs(etasub_term_safe[k].real) < 1e-15 and np.abs(etasub_term_safe[k].imag) < 1e-15):
                     inv_etasub_term_safe[k] = 1.0 / etasub_term_safe[k]
                 elif valid[k]:
                     valid[k] = False
             etasub = (nSub_complex**2) * inv_etasub_term_safe
    etainc_safe = np.empty_like(etainc)
    etasub_safe = np.empty_like(etasub)
    for k in range(N_angles):
        if not valid[k] or ~np.isfinite(etasub[k].real) or ~np.isfinite(etasub[k].imag):
             valid[k] = False
             etasub_safe[k] = 1.0 + 0j
        else:
             etasub_safe[k] = etasub[k]
        if not valid[k] or ~np.isfinite(etainc[k].real) or ~np.isfinite(etainc[k].imag):
            valid[k] = False
            etainc_safe[k] = 1.0 + 0j
        else:
            etainc_safe[k] = etainc[k]
    M11, M12, M21, M22 = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    num_r = (etainc_safe * M11 + etainc_safe * etasub_safe * M12 - M21 - etasub_safe * M22)
    den_r = (etainc_safe * M11 + etainc_safe * etasub_safe * M12 + M21 + etasub_safe * M22)
    r_infini = np.full(N_angles, np.nan + 1j*np.nan, dtype=np.complex128)
    for k in range(N_angles):
        den_is_zero = np.abs(den_r[k].real) < 1e-15 and np.abs(den_r[k].imag) < 1e-15
        den_is_finite = np.isfinite(den_r[k].real) and np.isfinite(den_r[k].imag)
        num_is_finite = np.isfinite(num_r[k].real) and np.isfinite(num_r[k].imag)
        if valid[k] and not den_is_zero and den_is_finite and num_is_finite:
            r_infini[k] = num_r[k] / den_r[k]
            if not (np.isfinite(r_infini[k].real) and np.isfinite(r_infini[k].imag)):
                 r_infini[k] = np.nan + 1j*np.nan
                 valid[k] = False
        else:
            valid[k] = False
    R_calc = np.abs(r_infini)**2
    R_final = np.full(N_angles, np.nan, dtype=np.float64)
    for k in range(N_angles):
        if valid[k] and np.isfinite(R_calc[k]):
             R_final[k] = min(max(R_calc[k], 0.0), 1.0)
    return R_final
@numba.njit(cache=True, fastmath=True)
def calcul_reflectance_angulaire_vectorized_numba(nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, num_layers):
    N_angles = len(angles_rad)
    nan_array = np.full(N_angles, np.nan, dtype=np.float64)
    if l0 <= 1e-12 or epaisseur <= 1e-12 or num_layers <= 0:
        return nan_array, nan_array
    if not (np.isfinite(nH_complex.real) and np.isfinite(nH_complex.imag) and
            np.isfinite(nL_complex.real) and np.isfinite(nL_complex.imag) and
            np.isfinite(nSub_complex.real) and np.isfinite(nSub_complex.imag)):
        return nan_array, nan_array
    if (nH_complex.real < -1e-9 or nL_complex.real < -1e-9 or nSub_complex.real < -1e-9 or
        nH_complex.imag < -1e-9 or nL_complex.imag < -1e-9 or nSub_complex.imag < -1e-9):
        return nan_array, nan_array
    n_inc = 1.0 + 0.0j
    alpha = n_inc * np.sin(angles_rad.astype(np.complex128))
    cos_theta = np.cos(angles_rad.astype(np.complex128))
    cos_theta_safe_p = np.empty_like(cos_theta)
    for i in range(N_angles):
        if np.abs(cos_theta[i].real) < 1e-15 and np.abs(cos_theta[i].imag) < 1e-15:
            cos_theta_safe_p[i] = 1e-15 + 0j
        else:
            cos_theta_safe_p[i] = cos_theta[i]
    Rs_final = _calculate_polarization_numba(True, nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, alpha, cos_theta, cos_theta_safe_p, num_layers)
    Rp_final = _calculate_polarization_numba(False, nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, alpha, cos_theta, cos_theta_safe_p, num_layers)
    return Rs_final, Rp_final
def objective_function_caller(params, nSub, l0, target_angles_rad, num_layers, angle_weights):
    rh, ch, rl, cl, epaisseur = params
    nH = rh + 1j * ch
    nL = rl + 1j * cl
    target_R = 0.0
    mse_total = 1e10
    try:
        Rs, Rp = calcul_reflectance_angulaire_vectorized_numba(
            np.complex128(nH), np.complex128(nL), np.complex128(nSub),
            l0, epaisseur, target_angles_rad, num_layers
        )
        all_nan_s = np.all(np.isnan(Rs))
        all_nan_p = np.all(np.isnan(Rp))
        if all_nan_s and all_nan_p:
            return 1e10
        Rs_filled = np.nan_to_num(Rs, nan=1.0)
        Rp_filled = np.nan_to_num(Rp, nan=1.0)
        se_s = (Rs_filled - target_R)**2
        se_p = (Rp_filled - target_R)**2
        if len(angle_weights) != len(target_angles_rad):
             return 1e10
        weighted_se_s = np.sum(angle_weights * se_s)
        weighted_se_p = np.sum(angle_weights * se_p)
        total_weight = 2 * np.sum(angle_weights)
        if total_weight <= 1e-9:
             valid_s = ~np.isnan(Rs)
             valid_p = ~np.isnan(Rp)
             valid_count = np.sum(valid_s) + np.sum(valid_p)
             if valid_count == 0:
                 return 1e10
             else:
                 mse_total = (np.sum(se_s[valid_s]) + np.sum(se_p[valid_p])) / valid_count
        else:
             mse_total = (weighted_se_s + weighted_se_p) / total_weight
        if not np.isfinite(mse_total) or mse_total < 0:
            mse_total = 1e10
    except (ValueError, OverflowError, FloatingPointError) as calc_err:
        return 1e10
    except Exception as e:
        return 1e10
    return mse_total
def get_complex_from_string(s, default=0j):
    try:
        return complex(s.replace(' ', '').replace('i', 'j'))
    except (ValueError, TypeError):
        return default
def format_complex(c):
    if isinstance(c, complex):
        return f"{c.real:.4f}{c.imag:+.4f}j"
    return str(c)
def format_complex_sigfig(c, n=6):
    if isinstance(c, complex) and np.isfinite(c.real) and np.isfinite(c.imag):
        real_part_str = f"{c.real:.{n}g}"
        imag_part = c.imag
        imag_part_str = f"{imag_part:.{n}g}"
        sign = "+" if imag_part >= 0 else ""
        return f"{real_part_str}{sign}{imag_part_str}j"
    elif isinstance(c, (int, float)) and np.isfinite(c):
         return f"{c:.{n}g}"
    return "N/A"
def format_bounds(bounds_list):
    if not bounds_list or len(bounds_list) != 5: return "N/A"
    names = ["rh", "ch", "rl", "cl", "ep"]
    parts = []
    for name, b in zip(names, bounds_list):
        low, high = b
        if name == "ep":
             parts.append(f"{name}:[{low:.1e},{high:.1e}]")
        else:
             parts.append(f"{name}:[{low:.2g},{high:.2g}]")
    return " ".join(parts)
@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    df_excel = df.copy()
    for col in df_excel.columns:
        if pd.api.types.is_complex_dtype(df_excel[col].dtype):
             df_excel[col] = df_excel[col].apply(lambda x: f"{x.real:.6g}{x.imag:+.6g}j" if pd.notna(x) and isinstance(x, complex) else x)
        elif df_excel[col].dtype == object:
             df_excel[col] = df_excel[col].apply(lambda x: f"{x.real:.6g}{x.imag:+.6g}j" if isinstance(x, complex) and pd.notna(x) else x)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_excel.to_excel(writer, index=False, sheet_name='OptimizationResults')
    processed_data = output.getvalue()
    return processed_data
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}
if 'final_plot_data' not in st.session_state:
    st.session_state.final_plot_data = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
st.set_page_config(layout="wide")
st.title("🔬 Optimisation Multicouche Anti-Reflet (Recuit Simulé)")
help_text = """
### ❓ Aide - Optimisation de Couche Anti-Reflet

**Objectif de l'Application**

Cette application vise à déterminer les propriétés optiques (indices de réfraction complexes $n_H$, $n_L$) et l'épaisseur ($e$) d'une structure périodique de $N$ couches minces alternées (type H-L-H-L...) déposées sur un substrat, afin de minimiser la réflectance pour une lumière incidente à une longueur d'onde ($\lambda$) donnée, sur une plage angulaire spécifiée. L'objectif est de concevoir un revêtement anti-reflet.

**Interface**

* **Barre Latérale (Gauche) :** Contient tous les paramètres d'entrée pour la simulation et l'optimisation.
* **Zone Principale (Droite) :** Affiche les logs, les résultats de l'optimisation et le tracé de la réflectance calculée.

**Paramètres d'Entrée (Barre Latérale)**

1.  **Paramètres Physiques :**
    * `Nombre de couches (N)` : Nombre total de couches H et L dans la structure (ex: N=5 -> H L H L H).
    * `a_substrat` : Paramètre lié à l'absorption du substrat. L'indice complexe du substrat est calculé comme $n_{Sub} = \sqrt{1 + i \cdot a_{sub}}$. Un $a_{sub} > 0$ implique un substrat absorbant.
    * `λ (l0) (µm)` : Longueur d'onde de la lumière incidente dans le vide, en micromètres.

2.  **Cible MSE :** Définit la plage angulaire sur laquelle la réflectance doit être minimisée.
    * `Angle début (°)` : Angle d'incidence minimum (par rapport à la normale) pour le calcul de l'erreur.
    * `Angle fin (°)` : Angle d'incidence maximum.
    * `Pas angle (°)` : Incrément angulaire entre le début et la fin.
    * *Pondération* : Une pondération linéaire est appliquée aux angles : l'erreur à `Angle fin` compte deux fois plus que l'erreur à `Angle début`. Cela force l'optimiseur à mieux performer aux angles plus élevés.

3.  **Paramètres d'Optimisation (Dual Annealing) :** Contrôle l'algorithme d'optimisation globale `scipy.optimize.dual_annealing`.
    * `Max Iterations (global)` : Nombre maximum d'itérations pour la phase de recherche globale (recuit simulé). Augmenter améliore les chances de trouver un bon minimum mais augmente le temps de calcul.
    * `Température Initiale` : Température de départ pour le recuit.
    * `Visit Parameter`, `Accept Parameter` : Paramètres contrôlant les probabilités d'exploration et d'acceptation de nouvelles solutions pendant le recuit.
    * `Tolérance (local search)` : Critère d'arrêt pour la phase de recherche locale (L-BFGS-B) qui affine la solution après le recuit global.
    * `Désactiver Recherche Locale` : Si coché, saute l'étape de recherche locale. Peut accélérer l'optimisation mais potentiellement au détriment de la précision finale de la solution.

4.  **Bornes d'Optimisation :** Définissent l'espace de recherche pour les paramètres inconnus.
    * `Indice Réel Min/Max (n H/L)` : Limites pour la partie réelle ($n$) des indices $n_H$ et $n_L$.
    * `Indice Imag Min/Max (k H/L)` : Limites pour la partie imaginaire ($k$) des indices $n_H$ et $n_L$. $k \ge 0$ correspond à l'absorption.
    * `Épaisseur Min/Max (µm)` : Limites pour l'épaisseur *individuelle* $e$ de chaque couche (supposée identique pour H et L dans ce modèle).

**Processus (Clic sur "Démarrer Optimisation")**

1.  **Validation :** Vérification de la cohérence des paramètres saisis.
2.  **Optimisation :** Lancement de l'algorithme `dual_annealing`.
    * Celui-ci tente de trouver les valeurs de $(n_H, n_L, e)$ (représentées par leurs parties réelle et imaginaire) à l'intérieur des bornes spécifiées, qui minimisent la fonction objectif.
    * **Fonction Objectif :** Calcule l'erreur quadratique moyenne (Mean Squared Error - MSE) pondérée entre la réflectance calculée et la cible (0) sur la plage angulaire définie. La réflectance est calculée pour les deux polarisations (Rs : S, Rp : P) en utilisant la **Méthode des Matrices de Transfert (TMM)**, accélérée via Numba.
        $MSE = \\frac{1}{2 \\sum w_i} \\sum_{i} w_i \\left[ (R_{S}(\\theta_i) - 0)^2 + (R_{P}(\\theta_i) - 0)^2 \\right]$
        où $w_i$ est le poids pour l'angle $\\theta_i$.
3.  **Affichage :** Les logs, résultats et le graphique final sont mis à jour.

**Sorties**

* **Logs :** Affiche les messages de progression, les paramètres finaux trouvés ($n_H = n+ik$, $n_L = n+ik$, $\epsilon_H = \epsilon'+i\epsilon''$, $\epsilon_L = \epsilon'+i\epsilon''$, Épaisseur) et le statut final. Note : $\epsilon = n^2$.
* **Résumé des Résultats :** Métriques clés : statut, durée, nombre d'évaluations, MSE final, et les valeurs optimisées pour $n_H, n_L, \epsilon_H, \epsilon_L$, épaisseur. Affiché avec une police plus petite.
* **Tracé de Réflectance :** Courbes de réflectance $R_S$ (bleu, continu) et $R_P$ (rouge, tirets) en fonction de l'angle d'incidence. L'axe Y est en échelle logarithmique pour mieux voir les faibles réflectances. La zone cible MSE est indiquée par un fond grisé. L'axe X s'arrête à l'angle fin spécifié.
* **Téléchargement Excel :** Permet de sauvegarder tous les paramètres d'entrée et les résultats détaillés de l'optimisation dans un fichier `.xlsx`.

**Conseils et Problèmes Courants**

* **Vitesse :** Le calcul peut être long. Numba accélère déjà fortement les calculs physiques. Pour aller plus vite :
    * Réduire `Max Iterations` (compromis sur la qualité).
    * Augmenter le `Pas angle (°)` pour la cible MSE (compromis sur la représentativité angulaire).
    * Essayer de cocher `Désactiver Recherche Locale` (compromis sur la précision finale).
    * Utiliser un ordinateur plus puissant.
* **Convergence :** Le `dual_annealing` est stochastique ; relancer l'optimisation peut donner des résultats légèrement différents. Si le MSE final est élevé ou si le statut est "Échec/Interrompu", essayez d'augmenter `Max Iterations` ou de revoir les bornes.
* **Première Exécution :** Le premier lancement après modification du code ou dans un nouvel environnement peut être plus lent car Numba compile les fonctions (`@numba.njit(cache=True)` utilise ensuite le cache).
* **Police des Résultats :** La taille de police réduite est obtenue via CSS. Si elle ne semble pas correcte, cela pourrait être dû à une mise à jour de Streamlit affectant les sélecteurs CSS.
"""
small_font_css = """
<style>
/* Cibler les conteneurs de st.metric */
div[data-testid="stMetric"] {
    /* Ajustements globaux si nécessaire */
}
/* Cibler le label (texte au-dessus de la valeur) */
div[data-testid="stMetric"] > label[data-testid="stMetricLabel"] > div {
    font-size: 0.85rem !important; /* Taille réduite pour le label */
    font-weight: 600 !important; /* Un peu plus gras */
    /* color: #555 !important; */ /* Optionnel: couleur plus discrète */
}
/* Cibler la valeur principale */
div[data-testid="stMetric"] > div[data-testid="stMetricValue"] {
    font-size: 0.95rem !important; /* Taille réduite pour la valeur */
    /* line-height: 1.2 !important; */ /* Optionnel: ajuster l'espacement */
}
/* Cibler la valeur delta (si elle était utilisée) */
/* div[data-testid="stMetric"] > div[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
} */
</style>
"""
with st.sidebar:
    st.header("Paramètres")
    with st.expander("**Paramètres Physiques**", expanded=True):
        num_layers = st.number_input("Nombre de couches (N)", min_value=1, step=1, value=5, key="num_layers")
        a_substrat_str = st.text_input("a_substrat (partie imag. de ε_sub)", value="5.0", key="a_substrat")
        l0_um = st.number_input("λ (l0) (µm)", min_value=0.01, step=0.01, value=1.0, format="%.3f", key="l0")
        try:
            a_sub = float(a_substrat_str)
            if a_sub >= 0:
                nSub_calc = np.sqrt(1 + 1j * a_sub)
                st.write(f"nSub (calculé): `{format_complex(nSub_calc)}`")
            else:
                st.warning("a_substrat doit être >= 0.")
                nSub_calc = None
        except ValueError:
            st.warning("a_substrat invalide.")
            nSub_calc = None
    with st.expander("**Cible MSE (Pondération Linéaire 1x->2x)**", expanded=True):
        mse_angle_start = st.number_input("Angle début (°)", min_value=0.0, max_value=89.9, step=1.0, value=0.0, key="mse_start")
        mse_angle_stop = st.number_input("Angle fin (°)", min_value=0.0, max_value=89.9, step=1.0, value=45.0, key="mse_stop")
        mse_angle_step = st.number_input("Pas angle (°)", min_value=0.1, step=0.1, value=5.0, key="mse_step")
        target_angles_deg_calc = np.array([])
        angle_weights_calc = np.array([])
        if mse_angle_step > 0 and mse_angle_stop >= mse_angle_start:
            target_angles_deg_calc = np.arange(mse_angle_start, mse_angle_stop + mse_angle_step * 0.5, mse_angle_step)
            if len(target_angles_deg_calc) > 0:
                 angles = target_angles_deg_calc
                 if len(angles) == 1:
                     angle_weights_calc = np.array([1.0])
                 else:
                     angle_start = angles[0]
                     angle_stop = angles[-1]
                     if np.isclose(angle_stop, angle_start):
                         angle_weights_calc = np.ones_like(angles)
                     else:
                         angle_weights_calc = 1.0 + (angles - angle_start) / (angle_stop - angle_start)
                 st.write(f"Angles MSE: `{len(target_angles_deg_calc)}` points de {mse_angle_start}° à {angles[-1]:.1f}°")
            else:
                 st.warning("Plage d'angles MSE invalide.")
        else:
            st.warning("Vérifiez les paramètres d'angle MSE.")
    with st.expander("**Paramètres d'Optimisation (Dual Annealing)**", expanded=False):
        max_iter = st.number_input("Max Iterations (global)", min_value=10, step=100, value=1000, key="maxiter")
        initial_temp = st.number_input("Température Initiale", min_value=1.0, value=5230.0, format="%.2f", key="temp")
        visit = st.number_input("Visit Parameter", value=2.62, format="%.2f", key="visit")
        accept = st.number_input("Accept Parameter", value=-5.0, format="%.2f", key="accept")
        tol = st.number_input("Tolérance (local search)", min_value=1e-9, value=1e-4, format="%.1e", key="tol")
        no_local_search = st.checkbox("Désactiver Recherche Locale", value=False, key="no_local")
    with st.expander("**Bornes d'Optimisation**", expanded=False):
        min_real = st.number_input("Indice Réel Min (n H/L)", min_value=0.01, value=0.1, format="%.3f", key="min_r")
        min_imag = st.number_input("Indice Imag Min (k H/L)", min_value=0.0, value=0.0, format="%.3f", key="min_i")
        max_index = st.number_input("Indice (n ou k) Max (H/L)", min_value=0.1, value=10.0, format="%.2f", key="max_idx")
        min_epaisseur = st.number_input("Épaisseur Min (µm)", min_value=1e-6, value=1e-4, format="%.1e", key="min_ep")
        max_epaisseur = st.number_input("Épaisseur Max (µm)", min_value=1e-5, value=0.1, format="%.1e", key="max_ep")
        bounds_calc = [
            (min_real, max_index), (min_imag, max_index),
            (min_real, max_index), (min_imag, max_index),
            (min_epaisseur, max_epaisseur)
        ]
        st.write(f"Bornes: `{format_bounds(bounds_calc)}`")
    st.divider()
    start_button = st.button("🚀 Démarrer Optimisation", type="primary")
    st.divider()
    with st.expander("❓ Aide / Informations"):
        st.markdown(help_text, unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📈 Statut et Résultats")
    log_placeholder = st.empty()
    results_placeholder = st.empty()
with col2:
    st.subheader("📊 Tracé de Réflectance")
    plot_placeholder = st.empty()
if start_button:
    st.session_state.optimization_results = None
    st.session_state.log_messages = ["Initialisation..."]
    st.session_state.final_plot_data = None
    st.session_state.result_df = None
    log_placeholder.text_area("Logs", "".join(st.session_state.log_messages), height=200, key="log_area_running")
    plot_placeholder.empty()
    results_placeholder.empty()
    valid_params = True
    if nSub_calc is None:
        st.sidebar.error("Erreur: nSub n'a pas pu être calculé (vérifiez a_substrat).")
        valid_params = False
    if len(target_angles_deg_calc) == 0:
        st.sidebar.error("Erreur: Aucun angle cible MSE valide.")
        valid_params = False
    if not (max_index > min_real and max_index > min_imag):
         st.sidebar.error("Erreur: Indice Max doit être supérieur aux Indices Min.")
         valid_params = False
    if not (max_epaisseur > min_epaisseur):
         st.sidebar.error("Erreur: Épaisseur Max doit être supérieure à Épaisseur Min.")
         valid_params = False
    if valid_params:
        params_run = {
            'num_layers': num_layers,
            'a_substrat': a_sub,
            'nSub': nSub_calc,
            'l0': l0_um,
            'mse_target_angles_deg': target_angles_deg_calc,
            'mse_target_angles_rad': np.radians(target_angles_deg_calc),
            'angle_weights': angle_weights_calc,
            'maxiter': max_iter,
            'initial_temp': initial_temp,
            'visit': visit,
            'accept': accept,
            'tol': tol,
            'no_local_search': no_local_search,
            'bounds': bounds_calc
        }
        st.session_state.last_params = params_run
        optimizer_args = (params_run['nSub'], params_run['l0'], params_run['mse_target_angles_rad'], params_run['num_layers'], params_run['angle_weights'])
        iteration_tracker = [0]
        def optimization_callback_st(xk, f, context):
            iteration_tracker[0] += 1
            current_iter = iteration_tracker[0]
            if current_iter % 10 == 0:
                 phase_str = ""
                 if context == 1: phase_str = "(Annealing)"
                 elif context == 2: phase_str = "(Local Search)"
                 log_msg = f"Callback {current_iter} {phase_str}... MSE={f:.3e}"
                 st.session_state.log_messages.append(log_msg + "\n")
        script_start_time = time.time()
        with st.status("⏳ Optimisation (Dual Annealing) en cours...", expanded=True) as status:
            st.write(f"Paramètres: N={params_run['num_layers']}, λ={params_run['l0']}µm, nSub={format_complex(params_run['nSub'])}")
            st.write(f"Angles Cible: {params_run['mse_target_angles_deg'][0]}° à {params_run['mse_target_angles_deg'][-1]}° (pas {mse_angle_step}°)")
            st.write(f"MaxIter={params_run['maxiter']}, Temp={params_run['initial_temp']:.1f}")
            st.session_state.log_messages.append("Optimisation démarrée...\n")
            result = None
            error_msg = None
            opt_duration = 0
            try:
                try:
                    dummy_p = [(b[0] + b[1]) / 2 for b in params_run['bounds']]
                    _ = objective_function_caller(dummy_p, params_run['nSub'], params_run['l0'], params_run['mse_target_angles_rad'][:2], params_run['num_layers'], params_run['angle_weights'][:2])
                    st.session_state.log_messages.append("Précompilation Numba OK.\n")
                except Exception as precompile_err:
                    st.session_state.log_messages.append(f"Avertissement précompilation: {precompile_err}\n")
                opt_start_time = time.time()
                result = dual_annealing(
                    func=objective_function_caller,
                    bounds=params_run['bounds'],
                    args=optimizer_args,
                    maxiter=params_run['maxiter'],
                    initial_temp=params_run['initial_temp'],
                    visit=params_run['visit'],
                    accept=params_run['accept'],
                    minimizer_kwargs={'tol': params_run['tol']},
                    no_local_search=params_run['no_local_search'],
                    callback=optimization_callback_st
                )
                opt_end_time = time.time()
                opt_duration = opt_end_time - opt_start_time
                st.session_state.log_messages.append(f"Optimisation terminée (durée: {opt_duration:.2f}s).\n")
            except Exception as e:
                opt_duration = time.time() - opt_start_time if 'opt_start_time' in locals() else 0
                error_msg = f"Erreur pendant l'optimisation: {e}"
                st.session_state.log_messages.append(f"ERREUR: {error_msg}\n")
                status.update(label="Optimisation échouée!", state="error", expanded=True)
            st.session_state.optimization_results = result
            final_status = "Erreur" if error_msg else ("Succès" if result and result.success else "Échec/Interrompu")
            final_message = error_msg if error_msg else (result.message if result else "N/A")
            results_dict = {
                "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Algorithm": "Dual Annealing",
                "Status": final_status,
                "Message": final_message,
                "Num_Layers": params_run.get('num_layers', 'N/A'),
                "a_Substrat": params_run.get('a_substrat', 'N/A'),
                "nSub": params_run.get('nSub', np.nan+0j),
                "Lambda_l0": params_run.get('l0', 'N/A'),
                "MSE_Angles_Deg": str(params_run.get('mse_target_angles_deg', 'N/A')),
                "MSE_Weights_Used": True,
                "Max_Iter": params_run.get('maxiter', 'N/A'),
                "Initial_Temp": params_run.get('initial_temp', 'N/A'),
                "Visit": params_run.get('visit', 'N/A'),
                "Accept": params_run.get('accept', 'N/A'),
                "Tolerance": params_run.get('tol', 'N/A'),
                "No_Local_Search": params_run.get('no_local_search', 'N/A'),
                "Bounds": format_bounds(params_run.get('bounds')),
                "Opt_Duration_s": f"{opt_duration:.2f}",
                "NFEV": result.nfev if result else 'N/A',
                "NIT_Global": result.nit if result else 'N/A',
                "Best_MSE_Weighted": np.nan,
                "nH": np.nan + 0j,
                "nL": np.nan + 0j,
                "Epaisseur_um": np.nan,
                "Epsilon_h": np.nan + 0j,
                "Epsilon_l": np.nan + 0j
            }
            best_params_tuple = None
            best_mse = np.inf
            final_best_nH = np.nan + 0j
            final_best_nL = np.nan + 0j
            final_best_epaisseur = np.nan
            epsilon_h = np.nan + 0j
            epsilon_l = np.nan + 0j
            if result and result.x is not None and np.isfinite(result.fun):
                 best_params_tuple = result.x
                 best_mse = result.fun
                 results_dict["Best_MSE_Weighted"] = f"{best_mse:.6e}"
                 st.session_state.log_messages.append(f"Statut final: {final_status} - MSE = {best_mse:.6e}\n")
                 best_rh, best_ch, best_rl, best_cl, final_best_epaisseur = best_params_tuple
                 final_best_nH = best_rh + 1j * best_ch
                 final_best_nL = best_rl + 1j * best_cl
                 epsilon_h = final_best_nH**2
                 epsilon_l = final_best_nL**2
                 results_dict.update({
                    "nH": final_best_nH,
                    "nL": final_best_nL,
                    "Epaisseur_um": final_best_epaisseur,
                    "Epsilon_h": epsilon_h,
                    "Epsilon_l": epsilon_l
                 })
                 st.session_state.log_messages.append(f"  nH = {format_complex(final_best_nH)}\n")
                 st.session_state.log_messages.append(f"  nL = {format_complex(final_best_nL)}\n")
                 st.session_state.log_messages.append(f"  εH = {format_complex(epsilon_h)}\n")
                 st.session_state.log_messages.append(f"  εL = {format_complex(epsilon_l)}\n")
                 st.session_state.log_messages.append(f"  Épaisseur = {final_best_epaisseur:.6f} µm\n")
                 try:
                    plot_angles_deg_fine = np.linspace(0, 90, 181)
                    plot_angles_rad_fine = np.radians(plot_angles_deg_fine)
                    final_Rs, final_Rp = calcul_reflectance_angulaire_vectorized_numba(
                        final_best_nH, final_best_nL, params_run['nSub'], params_run['l0'], final_best_epaisseur,
                        plot_angles_rad_fine, params_run['num_layers']
                    )
                    st.session_state.final_plot_data = {
                        "angles_deg": plot_angles_deg_fine,
                        "Rs": final_Rs, "Rp": final_Rp,
                        "nH": final_best_nH, "nL": final_best_nL, "ep": final_best_epaisseur,
                        "nSub": params_run['nSub'], "l0": params_run['l0'], "N": params_run['num_layers'],
                        "mse": best_mse, "status": final_status
                    }
                 except Exception as plot_err:
                    st.session_state.log_messages.append(f"Erreur pendant la préparation du tracé final: {plot_err}\n")
                    st.session_state.final_plot_data = None
            else:
                 st.session_state.log_messages.append(f"Statut final: {final_status} - Aucun résultat utilisable trouvé.\n")
                 results_dict["Best_MSE_Weighted"] = "N/A"
            if final_status == "Succès":
                 status.update(label=f"Optimisation Réussie! MSE={best_mse:.3e}", state="complete", expanded=False)
            else:
                 status.update(label=f"Optimisation Terminée ({final_status})", state="error" if final_status=="Erreur" else "complete", expanded=True)
            st.session_state.result_df = pd.DataFrame([results_dict])
    else:
        st.error("Veuillez corriger les erreurs dans les paramètres avant de lancer l'optimisation.")
        st.session_state.optimization_results = None
        st.session_state.log_messages = ["Erreur de paramètres détectée."]
        st.session_state.final_plot_data = None
        st.session_state.result_df = None
log_placeholder.text_area("Logs", "".join(st.session_state.log_messages), height=200, key="log_area_final", disabled=True)
if st.session_state.result_df is not None:
    st.markdown(small_font_css, unsafe_allow_html=True)
    with results_placeholder.container():
        st.subheader("Résumé des Résultats")
        df_display = st.session_state.result_df.iloc[0]
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Statut", df_display.get("Status", "N/A"))
            st.metric("Durée (s)", df_display.get("Opt_Duration_s", "N/A"))
            st.metric("NFEV (Évals Func.)", df_display.get("NFEV", "N/A"))
            st.metric("NIT (Itér. Globales)", df_display.get("NIT_Global", "N/A"))
        with res_col2:
            st.metric("Meilleur MSE Pondéré", df_display.get("Best_MSE_Weighted", "N/A"))
            nh_val = df_display.get('nH', np.nan + 0j)
            nl_val = df_display.get('nL', np.nan + 0j)
            st.metric("nH (n+ik)", format_complex_sigfig(nh_val, 6))
            st.metric("nL (n+ik)", format_complex_sigfig(nl_val, 6))
            st.metric("Épaisseur (µm)", f"{df_display.get('Epaisseur_um', np.nan):.6g}" if pd.notna(df_display.get('Epaisseur_um')) else "N/A")
        with res_col3:
             st.metric("Couches (N)", f"{df_display.get('Num_Layers', 'N/A')}")
             st.metric("λ (µm)", f"{df_display.get('Lambda_l0', np.nan):.3f}" if pd.notna(df_display.get('Lambda_l0')) else "N/A")
             eps_h_val = df_display.get('Epsilon_h', np.nan + 0j)
             eps_l_val = df_display.get('Epsilon_l', np.nan + 0j)
             st.metric("εH (ε'+iε'')", format_complex_sigfig(eps_h_val, 6))
             st.metric("εL (ε'+iε'')", format_complex_sigfig(eps_l_val, 6))
        with st.expander("Voir tous les détails des paramètres et résultats"):
            st.dataframe(st.session_state.result_df.astype(str).T)
        excel_data = convert_df_to_excel(st.session_state.result_df)
        st.download_button(
            label="📥 Télécharger les résultats (Excel)",
            data=excel_data,
            file_name=f"optimization_results_SA_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
if st.session_state.final_plot_data is not None:
    with plot_placeholder.container():
        plot_data = st.session_state.final_plot_data
        fig, ax = plt.subplots(figsize=(7, 5))
        valid_s = ~np.isnan(plot_data["angles_deg"]) & ~np.isnan(plot_data["Rs"])
        valid_p = ~np.isnan(plot_data["angles_deg"]) & ~np.isnan(plot_data["Rp"])
        ax.plot(plot_data["angles_deg"][valid_s], plot_data["Rs"][valid_s], label='Rs', linestyle='-', color='blue', marker='.', markersize=3)
        ax.plot(plot_data["angles_deg"][valid_p], plot_data["Rp"][valid_p], label='Rp', linestyle='--', color='red', marker='x', markersize=3)
        xlim_upper = 90.0
        if 'last_params' in st.session_state and st.session_state.last_params and 'mse_target_angles_deg' in st.session_state.last_params:
             mse_angles = st.session_state.last_params['mse_target_angles_deg']
             if len(mse_angles) > 0:
                  ax.axvspan(mse_angles[0], mse_angles[-1], color='lightgray', alpha=0.3, label='Zone Cible MSE')
                  xlim_upper = mse_angles[-1]
        ax.set_xlabel("Angle d'incidence (degrés)")
        ax.set_ylabel("Réflectance (échelle log)")
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize='small')
        ax.set_ylim(1e-6, 1.1)
        ax.set_xlim(0, max(1.0, xlim_upper))
        nH_str = format_complex_sigfig(plot_data['nH'], 4)
        nL_str = format_complex_sigfig(plot_data['nL'], 4)
        nSub_str = format_complex_sigfig(plot_data['nSub'], 4)
        ep_str = f"{plot_data['ep']:.4g}"
        mse_str = f"{plot_data['mse']:.3e}"
        l0_str = f"{plot_data['l0']:.3f}"
        status_str = plot_data['status']
        title = (f"Résultat Final ({status_str}) | MSE_pondéré: {mse_str} | λ={l0_str}µm\n"
                 f"nH={nH_str}, nL={nL_str}, ép={ep_str}µm | nSub={nSub_str}, N={plot_data['N']}")
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
else:
     with plot_placeholder.container():
          st.info("Le tracé de réflectance apparaîtra ici après une optimisation réussie.")
