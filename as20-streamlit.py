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
import io # Pour le téléchargement Excel

# --- Configuration et Fonctions Numba (Identiques à l'original) ---
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)

@numba.njit(cache=True, fastmath=True)
def _calculate_polarization_numba(pol_type_is_s, nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, alpha, cos_theta, cos_theta_safe_p, num_layers):
    # ... (votre code _calculate_polarization_numba complet et inchangé ici) ...
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
            # Utiliser ~np.isclose pour une meilleure lisibilité et potentiellement Numba-friendly
            # if np.isclose(term_sqrt[k].real, 0.0) and np.isclose(term_sqrt[k].imag, 0.0) or not np.isfinite(term_sqrt[k].real) or not np.isfinite(term_sqrt[k].imag):
            if ~np.isfinite(term_sqrt[k].real) or ~np.isfinite(term_sqrt[k].imag) or (np.abs(term_sqrt[k].real) < 1e-15 and np.abs(term_sqrt[k].imag) < 1e-15):
                valid[k] = False
                term_sqrt_safe[k] = 1.0 + 0j # Donner une valeur sûre pour éviter les erreurs en aval
            else:
                term_sqrt_safe[k] = term_sqrt[k]

        if pol_type_is_s:
            eta = term_sqrt_safe # Pour s, eta = n * cos(theta_i) = sqrt(n^2 - alpha^2)
        else: # Cas polarisation p
            # Vérifier si Ni est proche de zéro
            # if np.isclose(Ni.real, 0.0) and np.isclose(Ni.imag, 0.0):
            if np.abs(Ni.real) < 1e-15 and np.abs(Ni.imag) < 1e-15:
                valid[:] = False # Si Ni est zéro, eta_p devient indéfini/infini
                eta = np.zeros(N_angles, dtype=np.complex128) # Ou une autre valeur non valide
            else:
                eta_denom_safe = np.empty_like(term_sqrt_safe)
                for k in range(N_angles):
                     # if valid[k] and not (np.isclose(term_sqrt_safe[k].real, 0.0) and np.isclose(term_sqrt_safe[k].imag, 0.0)):
                     if valid[k] and not (np.abs(term_sqrt_safe[k].real) < 1e-15 and np.abs(term_sqrt_safe[k].imag) < 1e-15):
                         eta_denom_safe[k] = term_sqrt_safe[k]
                     elif valid[k]: # Si term_sqrt est zéro mais on était valide, invalider
                         valid[k] = False
                         eta_denom_safe[k] = 1.0 + 0j # Valeur sûre
                     else: # Déjà invalide
                         eta_denom_safe[k] = 1.0 + 0j # Valeur sûre

                inv_eta_denom_safe = np.zeros_like(eta_denom_safe)
                for k in range(N_angles):
                    # if valid[k] and not (np.isclose(eta_denom_safe[k].real, 0.0) and np.isclose(eta_denom_safe[k].imag, 0.0)):
                    if valid[k] and not (np.abs(eta_denom_safe[k].real) < 1e-15 and np.abs(eta_denom_safe[k].imag) < 1e-15):
                        inv_eta_denom_safe[k] = 1.0 / eta_denom_safe[k]
                    elif valid[k]: # Si denom est zéro, invalider
                        valid[k] = False

                # eta_p = n^2 / (n * cos(theta_i)) = n^2 / sqrt(n^2 - alpha^2)
                eta = (Ni**2) * inv_eta_denom_safe

        # Vérifier eta et assigner eta_safe
        for k in range(N_angles):
             # if not valid[k] or (np.isclose(eta[k].real, 0.0) and np.isclose(eta[k].imag, 0.0)) or not np.isfinite(eta[k].real) or not np.isfinite(eta[k].imag):
             if not valid[k] or ~np.isfinite(eta[k].real) or ~np.isfinite(eta[k].imag) or (np.abs(eta[k].real) < 1e-15 and np.abs(eta[k].imag) < 1e-15):
                 valid[k] = False
                 eta_safe[k] = 1.0 + 0j # Valeur sûre
             else:
                 eta_safe[k] = eta[k]


        phi = (2 * np.pi / l0) * term_sqrt_safe * epi # Phase accumulée
        # Utiliser des opérations in-place pour cos/sin si Numba le permet bien
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Calculer inv_eta_safe pour Mi[0, 1]
        inv_eta_safe = np.zeros_like(eta_safe)
        for k in range(N_angles):
            # if valid[k] and not (np.isclose(eta_safe[k].real, 0.0) and np.isclose(eta_safe[k].imag, 0.0)):
            if valid[k] and not (np.abs(eta_safe[k].real) < 1e-15 and np.abs(eta_safe[k].imag) < 1e-15):
                inv_eta_safe[k] = 1.0 / eta_safe[k]
            elif valid[k]: # eta_safe est zéro -> division par zéro -> invalide
                valid[k] = False
                # inv_eta_safe[k] restera 0.0+0.0j mais sera ignoré car valid[k] est False


        # Construire la matrice de couche Mi
        # Utiliser l'assignation directe si possible
        Mi[:, 0, 0] = cos_phi
        Mi[:, 0, 1] = 1j * inv_eta_safe * sin_phi
        Mi[:, 1, 0] = 1j * eta_safe * sin_phi
        Mi[:, 1, 1] = cos_phi

        # Multiplication matricielle M = Mi * M_prev
        # Stocker l'ancien M temporairement
        M_prev = M.copy() # Important car M est utilisé dans le calcul de M_new
        M_new = np.empty_like(M) # Créer une nouvelle matrice pour les résultats

        for k in range(N_angles):
            if not valid[k]:
                M_new[k] = M_prev[k] # Conserver l'ancienne valeur si invalide
                continue

            # Vérifier la finitude de Mi avant la multiplication
            mi_is_finite = (np.isfinite(Mi[k,0,0].real) and np.isfinite(Mi[k,0,0].imag) and
                              np.isfinite(Mi[k,0,1].real) and np.isfinite(Mi[k,0,1].imag) and
                              np.isfinite(Mi[k,1,0].real) and np.isfinite(Mi[k,1,0].imag) and
                              np.isfinite(Mi[k,1,1].real) and np.isfinite(Mi[k,1,1].imag))

            if not mi_is_finite:
                valid[k] = False
                M_new[k] = M_prev[k]
                continue

            # Multiplication
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

            # Vérifier la finitude du résultat de la multiplication
            m_new_is_finite = (np.isfinite(m00.real) and np.isfinite(m00.imag) and
                               np.isfinite(m01.real) and np.isfinite(m01.imag) and
                               np.isfinite(m10.real) and np.isfinite(m10.imag) and
                               np.isfinite(m11.real) and np.isfinite(m11.imag))

            if not m_new_is_finite:
                valid[k] = False
                M_new[k] = M_prev[k] # Revenir à la valeur précédente en cas de non-finitude

        M = M_new # Mettre à jour M pour la prochaine itération

    # --- Calcul final de la réflectance ---

    # Impédances optiques du milieu incident (air n=1) et du substrat
    eta_sub_term_sq = nSub_complex**2 - alpha**2
    etasub_term = np.sqrt(eta_sub_term_sq)

    etainc = np.empty(N_angles, dtype=np.complex128)
    etasub = np.empty(N_angles, dtype=np.complex128)

    if pol_type_is_s:
        etainc = cos_theta # eta_inc_s = n_inc * cos(theta_inc) = 1 * cos(theta)
        # Calcul de eta_sub_s = n_sub * cos(theta_sub) = sqrt(n_sub^2 - alpha^2)
        for k in range(N_angles):
            # if not valid[k] or not np.isfinite(etasub_term[k].real) or not np.isfinite(etasub_term[k].imag) or (np.isclose(etasub_term[k].real, 0.0) and np.isclose(etasub_term[k].imag, 0.0)):
            if not valid[k] or ~np.isfinite(etasub_term[k].real) or ~np.isfinite(etasub_term[k].imag):
                valid[k] = False
                etasub[k] = 1.0 + 0j # Valeur sûre
            # elif np.abs(etasub_term[k].real) < 1e-15 and np.abs(etasub_term[k].imag) < 1e-15:
            #     valid[k] = False # Si eta_sub est zéro, cela peut poser problème plus tard
            #     etasub[k] = 1.0 + 0j
            else:
                 etasub[k] = etasub_term[k]

    else: # Polarisation p
        # eta_inc_p = n_inc / cos(theta_inc) = 1 / cos(theta)
        inv_cos_theta_safe_p = np.zeros_like(cos_theta_safe_p)
        for k in range(N_angles):
            # Utiliser cos_theta_safe_p qui gère le cas cos(theta)=0
            # if valid[k] and not (np.isclose(cos_theta_safe_p[k].real, 0.0) and np.isclose(cos_theta_safe_p[k].imag, 0.0)):
            if valid[k] and not (np.abs(cos_theta_safe_p[k].real) < 1e-15 and np.abs(cos_theta_safe_p[k].imag) < 1e-15):
                 inv_cos_theta_safe_p[k] = 1.0 / cos_theta_safe_p[k]
            elif valid[k]:
                 valid[k] = False # Division par zéro si cos_theta_safe_p est proche de zéro
        etainc = inv_cos_theta_safe_p # Note: n_inc = 1 implicitement

        # eta_sub_p = n_sub^2 / (n_sub * cos(theta_sub)) = n_sub^2 / sqrt(n_sub^2 - alpha^2)
        # Vérifier si nSub est proche de zéro
        # if np.isclose(nSub_complex.real, 0.0) and np.isclose(nSub_complex.imag, 0.0):
        if np.abs(nSub_complex.real) < 1e-15 and np.abs(nSub_complex.imag) < 1e-15:
             valid[:] = False # nSub=0 rend eta_sub_p indéfini
             etasub[:] = 0.0 + 0j
        else:
             etasub_term_safe = np.empty_like(etasub_term)
             for k in range(N_angles):
                 # if not valid[k] or not np.isfinite(etasub_term[k].real) or not np.isfinite(etasub_term[k].imag) or (np.isclose(etasub_term[k].real, 0.0) and np.isclose(etasub_term[k].imag, 0.0)):
                 if not valid[k] or ~np.isfinite(etasub_term[k].real) or ~np.isfinite(etasub_term[k].imag) or (np.abs(etasub_term[k].real) < 1e-15 and np.abs(etasub_term[k].imag) < 1e-15):
                     valid[k] = False
                     etasub_term_safe[k] = 1.0 + 0j # Valeur sûre pour éviter division par zéro
                 else:
                     etasub_term_safe[k] = etasub_term[k]

             inv_etasub_term_safe = np.zeros_like(etasub_term_safe)
             for k in range(N_angles):
                 # if valid[k] and not (np.isclose(etasub_term_safe[k].real, 0.0) and np.isclose(etasub_term_safe[k].imag, 0.0)):
                 if valid[k] and not (np.abs(etasub_term_safe[k].real) < 1e-15 and np.abs(etasub_term_safe[k].imag) < 1e-15):
                     inv_etasub_term_safe[k] = 1.0 / etasub_term_safe[k]
                 elif valid[k]: # Division par zéro
                     valid[k] = False

             etasub = (nSub_complex**2) * inv_etasub_term_safe


    # Vérifications finales et valeurs sûres pour etainc et etasub
    etainc_safe = np.empty_like(etainc)
    etasub_safe = np.empty_like(etasub)
    for k in range(N_angles):
        # Vérifier etasub
        # if not valid[k] or not np.isfinite(etasub[k].real) or not np.isfinite(etasub[k].imag) or (np.isclose(etasub[k].real, 0.0) and np.isclose(etasub[k].imag, 0.0)):
        if not valid[k] or ~np.isfinite(etasub[k].real) or ~np.isfinite(etasub[k].imag):
             valid[k] = False
             etasub_safe[k] = 1.0 + 0j # Donner une valeur sûre
        # elif np.abs(etasub[k].real) < 1e-15 and np.abs(etasub[k].imag) < 1e-15:
             # Etasub = 0 peut être valide dans certains cas, mais peut causer des problèmes dans le dénominateur de r
             # Pour l'instant, on ne l'invalide pas ici, mais on vérifie den_r plus tard
             # valid[k] = False # Optionnel: invalider si etasub est zéro
             # etasub_safe[k] = 1.0 + 0j
        else:
             etasub_safe[k] = etasub[k]

        # Vérifier etainc
        # if not valid[k] or not np.isfinite(etainc[k].real) or not np.isfinite(etainc[k].imag) or (np.isclose(etainc[k].real, 0.0) and np.isclose(etainc[k].imag, 0.0)):
        if not valid[k] or ~np.isfinite(etainc[k].real) or ~np.isfinite(etainc[k].imag):
            valid[k] = False
            etainc_safe[k] = 1.0 + 0j # Valeur sûre
        # elif np.abs(etainc[k].real) < 1e-15 and np.abs(etainc[k].imag) < 1e-15:
            # Idem, etainc=0 peut arriver (angle 90 deg pour eta_s), vérifier den_r
            # valid[k] = False # Optionnel
            # etainc_safe[k] = 1.0 + 0j
        else:
            etainc_safe[k] = etainc[k]


    # Calcul du coefficient de réflexion r
    M11, M12, M21, M22 = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    num_r = (etainc_safe * M11 + etainc_safe * etasub_safe * M12 - M21 - etasub_safe * M22)
    den_r = (etainc_safe * M11 + etainc_safe * etasub_safe * M12 + M21 + etasub_safe * M22)

    # Initialiser r avec NaN complexe
    r_infini = np.full(N_angles, np.nan + 1j*np.nan, dtype=np.complex128)

    # Calculer r seulement si valide et dénominateur non nul et fini
    for k in range(N_angles):
        # den_is_zero = np.isclose(den_r[k].real, 0.0) and np.isclose(den_r[k].imag, 0.0)
        den_is_zero = np.abs(den_r[k].real) < 1e-15 and np.abs(den_r[k].imag) < 1e-15
        den_is_finite = np.isfinite(den_r[k].real) and np.isfinite(den_r[k].imag)
        num_is_finite = np.isfinite(num_r[k].real) and np.isfinite(num_r[k].imag)

        if valid[k] and not den_is_zero and den_is_finite and num_is_finite:
            r_infini[k] = num_r[k] / den_r[k]
            # Vérifier si le résultat de la division est fini
            if not (np.isfinite(r_infini[k].real) and np.isfinite(r_infini[k].imag)):
                 r_infini[k] = np.nan + 1j*np.nan # Invalider si la division donne inf/NaN
                 valid[k] = False # Marquer comme invalide globalement aussi
        else:
            # Si déjà invalide, ou division par zéro, ou non-fini, r reste NaN
            valid[k] = False # Assurer que c'est bien marqué comme invalide

    # Calcul de la Réflectance R = |r|^2
    # Utiliser np.abs() qui gère les complexes
    R_calc = np.abs(r_infini)**2

    # Initialiser R_final avec NaN
    R_final = np.full(N_angles, np.nan, dtype=np.float64)

    # Assigner R_final seulement si le calcul était valide et le résultat est fini
    for k in range(N_angles):
        if valid[k] and np.isfinite(R_calc[k]):
             # R ne peut pas être > 1 physiquement. Plafonner à 1.
             # Gérer les petites erreurs numériques qui pourraient dépasser 1.
             R_final[k] = min(max(R_calc[k], 0.0), 1.0)
        # else: R_final[k] reste NaN

    return R_final


@numba.njit(cache=True, fastmath=True)
def calcul_reflectance_angulaire_vectorized_numba(nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, num_layers):
    N_angles = len(angles_rad)
    nan_array = np.full(N_angles, np.nan, dtype=np.float64)

    # --- Vérifications initiales robustes ---
    # Paramètres physiques de base
    if l0 <= 1e-12 or epaisseur <= 1e-12 or num_layers <= 0: # Utiliser une petite tolérance
        return nan_array, nan_array

    # Validité des indices complexes (partie réelle >= 0, partie imaginaire >= 0)
    # Et finitude
    if not (np.isfinite(nH_complex.real) and np.isfinite(nH_complex.imag) and
            np.isfinite(nL_complex.real) and np.isfinite(nL_complex.imag) and
            np.isfinite(nSub_complex.real) and np.isfinite(nSub_complex.imag)):
        return nan_array, nan_array

    # Convention physique: partie réelle >= 0, partie imaginaire (liée à l'absorption) >= 0
    # Certains modèles peuvent permettre n_imag < 0 (gain), mais ici on suppose l'absorption/transparence.
    # On tolère une petite valeur négative pour la partie réelle due aux erreurs numériques, mais pas trop.
    if (nH_complex.real < -1e-9 or nL_complex.real < -1e-9 or nSub_complex.real < -1e-9 or
        nH_complex.imag < -1e-9 or nL_complex.imag < -1e-9 or nSub_complex.imag < -1e-9):
        return nan_array, nan_array

    # Si une partie imaginaire est négative (gain), on le signale peut-être mais on continue
    # Attention: Si n=0, cela peut causer des problèmes. Géré dans _calculate_polarization_numba


    # --- Préparation des entrées pour _calculate_polarization_numba ---
    # Milieu incident supposé être l'air (n_inc = 1.0 + 0.0j)
    n_inc = 1.0 + 0.0j

    # Calcul de alpha = n_inc * sin(theta_inc)
    # S'assurer que angles_rad sont dans [0, pi/2] - supposé ok par l'appelant
    # Utiliser np.complex128 pour la compatibilité Numba si nécessaire
    alpha = n_inc * np.sin(angles_rad.astype(np.complex128)) # alpha est réel si n_inc réel

    # Calcul de cos(theta_inc)
    cos_theta = np.cos(angles_rad.astype(np.complex128)) # cos_theta est réel

    # Créer une version "sûre" de cos_theta pour éviter la division par zéro dans eta_p
    cos_theta_safe_p = np.empty_like(cos_theta)
    for i in range(N_angles):
        # if np.isclose(cos_theta[i].real, 0.0) and np.isclose(cos_theta[i].imag, 0.0):
        if np.abs(cos_theta[i].real) < 1e-15 and np.abs(cos_theta[i].imag) < 1e-15: # Angle de 90 degrés
            # Donner une très petite valeur pour éviter la division par zéro,
            # tout en étant assez petit pour que le résultat soit potentiellement invalidé plus tard si nécessaire.
            cos_theta_safe_p[i] = 1e-15 + 0j
        else:
            cos_theta_safe_p[i] = cos_theta[i]


    # --- Appels à la fonction de calcul principale ---
    Rs_final = _calculate_polarization_numba(True, nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, alpha, cos_theta, cos_theta_safe_p, num_layers)
    Rp_final = _calculate_polarization_numba(False, nH_complex, nL_complex, nSub_complex, l0, epaisseur, angles_rad, alpha, cos_theta, cos_theta_safe_p, num_layers)

    return Rs_final, Rp_final


def objective_function_caller(params, nSub, l0, target_angles_rad, num_layers, angle_weights):
    # Dépaqueter les paramètres
    rh, ch, rl, cl, epaisseur = params
    nH = rh + 1j * ch
    nL = rl + 1j * cl
    target_R = 0.0 # Cible est une réflectance nulle (anti-reflet)

    # Initialiser la MSE à une grande valeur
    mse_total = 1e10 # Valeur par défaut en cas d'échec du calcul

    # --- Bloc try/except robuste ---
    try:
        # Appeler la fonction Numba vectorisée
        Rs, Rp = calcul_reflectance_angulaire_vectorized_numba(
            np.complex128(nH), np.complex128(nL), np.complex128(nSub),
            l0, epaisseur, target_angles_rad, num_layers
        )

        # --- Vérification des résultats de Numba ---
        # Si tous les résultats sont NaN, les paramètres sont probablement invalides
        all_nan_s = np.all(np.isnan(Rs))
        all_nan_p = np.all(np.isnan(Rp))

        if all_nan_s and all_nan_p:
            # st.write(f"DEBUG: All NaN for params: {params}") # Debug Streamlit
            return 1e10 # Retourner une grande pénalité

        # --- Calcul de l'erreur quadratique pondérée ---
        # Remplacer les NaN par une réflectance maximale (1.0) pour pénaliser
        # les angles où le calcul a échoué.
        Rs_filled = np.nan_to_num(Rs, nan=1.0)
        Rp_filled = np.nan_to_num(Rp, nan=1.0)

        # Erreur quadratique pour chaque polarisation et chaque angle
        se_s = (Rs_filled - target_R)**2
        se_p = (Rp_filled - target_R)**2

        # Assurer que angle_weights a la même taille que target_angles_rad
        if len(angle_weights) != len(target_angles_rad):
             # Ceci ne devrait pas arriver si les paramètres sont bien générés
             # Gérer l'erreur ou utiliser un poids uniforme
             # Pour l'instant, on retourne une pénalité forte
             # print("Warning: Mismatch between angle_weights and target_angles_rad length.")
             return 1e10

        # Somme pondérée des erreurs quadratiques
        weighted_se_s = np.sum(angle_weights * se_s)
        weighted_se_p = np.sum(angle_weights * se_p)

        # Poids total (pour la normalisation)
        # Chaque angle contribue pour les deux polarisations
        total_weight = 2 * np.sum(angle_weights)

        # Éviter la division par zéro si tous les poids sont nuls (cas improbable)
        if total_weight <= 1e-9:
             # Si pas de poids, calculer une MSE simple sur les points valides
             valid_s = ~np.isnan(Rs)
             valid_p = ~np.isnan(Rp)
             valid_count = np.sum(valid_s) + np.sum(valid_p)
             if valid_count == 0:
                 return 1e10 # Aucun point valide
             else:
                 # Prendre la moyenne des SE calculées sur les points valides
                 mse_total = (np.sum(se_s[valid_s]) + np.sum(se_p[valid_p])) / valid_count
        else:
             # MSE pondérée totale
             mse_total = (weighted_se_s + weighted_se_p) / total_weight


        # --- Vérification finale de la MSE ---
        # S'assurer que mse_total est un nombre fini et positif
        if not np.isfinite(mse_total) or mse_total < 0:
            # Tenter de diagnostiquer: est-ce qu'un SE pondéré était infini/NaN?
            # if np.any(~np.isfinite(angle_weights * se_s)) or np.any(~np.isfinite(angle_weights * se_p)):
            #    print(f"DEBUG: Non-finite weighted SE for params: {params}") # Debug local
            # else:
            #    print(f"DEBUG: Non-finite MSE calculation for params: {params}, MSE={mse_total}") # Debug local

            mse_total = 1e10 # Retourner une grande pénalité en cas de problème

    # --- Gestion des exceptions ---
    except (ValueError, OverflowError, FloatingPointError) as calc_err:
        # Capturer les erreurs numériques ou autres problèmes pendant le calcul
        # print(f"DEBUG: Calculation Error for params {params}: {calc_err}") # Debug local
        return 1e10 # Pénalité forte en cas d'erreur de calcul
    except Exception as e:
        # Capturer toute autre exception inattendue
        # print(f"DEBUG: Unexpected Error in objective function for params {params}: {e}") # Debug local
        # Il est préférable de logger l'erreur complète pour le débogage
        # import traceback
        # traceback.print_exc()
        return 1e10 # Pénalité forte

    # Retourner la MSE calculée (ou la pénalité si erreur)
    return mse_total

# --- Fonctions Utilitaires ---
def get_complex_from_string(s, default=0j):
    """Essaie de convertir une chaîne en nombre complexe."""
    try:
        return complex(s.replace(' ', '').replace('i', 'j'))
    except (ValueError, TypeError):
        return default

def format_complex(c):
    """Formate un nombre complexe pour l'affichage."""
    if isinstance(c, complex):
        return f"{c.real:.4f}{c.imag:+.4f}j"
    return str(c)

def format_bounds(bounds_list):
    """Formate les bornes pour l'affichage."""
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

# Fonction pour convertir DataFrame en Excel en mémoire pour téléchargement
@st.cache_data # Cache la conversion pour éviter de la refaire inutilement
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='OptimizationResults')
    processed_data = output.getvalue()
    return processed_data

# --- Initialisation de l'état Streamlit ---
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


# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("🔬 Optimisation Multicouche Anti-Reflet (Recuit Simulé)")

# --- Barre Latérale pour les Paramètres ---
with st.sidebar:
    st.header("Paramètres")

    # Section Paramètres Physiques
    with st.expander("**Paramètres Physiques**", expanded=True):
        num_layers = st.number_input("Nombre de couches (N)", min_value=1, step=1, value=5, key="num_layers")
        a_substrat_str = st.text_input("a_substrat (partie imag. de ε_sub)", value="5.0", key="a_substrat")
        l0_um = st.number_input("λ (l0) (µm)", min_value=0.01, step=0.01, value=1.0, format="%.3f", key="l0")

        # Calcul et affichage dynamique de nSub
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

    # Section Cible MSE
    with st.expander("**Cible MSE (Pondération Linéaire 1x->2x)**", expanded=True):
        mse_angle_start = st.number_input("Angle début (°)", min_value=0.0, max_value=89.9, step=1.0, value=0.0, key="mse_start")
        mse_angle_stop = st.number_input("Angle fin (°)", min_value=0.0, max_value=89.9, step=1.0, value=45.0, key="mse_stop")
        mse_angle_step = st.number_input("Pas angle (°)", min_value=0.1, step=0.1, value=5.0, key="mse_step")

        # Calcul des angles cibles et poids
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
                         # Pondération linéaire de 1 (début) à 2 (fin)
                         angle_weights_calc = 1.0 + (angles - angle_start) / (angle_stop - angle_start)
                 st.write(f"Angles MSE: `{len(target_angles_deg_calc)}` points de {mse_angle_start}° à {angles[-1]:.1f}°")
            else:
                 st.warning("Plage d'angles MSE invalide.")
        else:
            st.warning("Vérifiez les paramètres d'angle MSE.")


    # Section Paramètres d'Optimisation (Recuit Simulé)
    with st.expander("**Paramètres d'Optimisation (Dual Annealing)**", expanded=False):
        max_iter = st.number_input("Max Iterations (global)", min_value=10, step=100, value=1000, key="maxiter")
        initial_temp = st.number_input("Température Initiale", min_value=1.0, value=5230.0, format="%.2f", key="temp")
        visit = st.number_input("Visit Parameter", value=2.62, format="%.2f", key="visit")
        accept = st.number_input("Accept Parameter", value=-5.0, format="%.2f", key="accept")
        tol = st.number_input("Tolérance (local search)", min_value=1e-9, value=1e-4, format="%.1e", key="tol")
        no_local_search = st.checkbox("Désactiver Recherche Locale", value=False, key="no_local")

    # Section Bornes d'Optimisation
    with st.expander("**Bornes d'Optimisation**", expanded=False):
        min_real = st.number_input("Indice Réel Min (n H/L)", min_value=0.01, value=0.1, format="%.3f", key="min_r")
        min_imag = st.number_input("Indice Imag Min (k H/L)", min_value=0.0, value=0.0, format="%.3f", key="min_i")
        max_index = st.number_input("Indice (n ou k) Max (H/L)", min_value=0.1, value=10.0, format="%.2f", key="max_idx")
        min_epaisseur = st.number_input("Épaisseur Min (µm)", min_value=1e-6, value=1e-4, format="%.1e", key="min_ep")
        max_epaisseur = st.number_input("Épaisseur Max (µm)", min_value=1e-5, value=0.1, format="%.1e", key="max_ep")

        bounds_calc = [
            (min_real, max_index), (min_imag, max_index), # nH (real, imag)
            (min_real, max_index), (min_imag, max_index), # nL (real, imag)
            (min_epaisseur, max_epaisseur)             # epaisseur
        ]
        st.write(f"Bornes: `{format_bounds(bounds_calc)}`")

    # Bouton de démarrage
    st.divider()
    start_button = st.button("🚀 Démarrer Optimisation", type="primary")


# --- Zone Principale pour les Résultats et le Tracé ---
col1, col2 = st.columns([1, 1]) # Deux colonnes: Statut/Résultats et Tracé

with col1:
    st.subheader("📈 Statut et Résultats")
    log_placeholder = st.empty() # Pour afficher les logs dynamiquement
    results_placeholder = st.empty() # Pour afficher les résultats finaux

with col2:
    st.subheader("📊 Tracé de Réflectance")
    plot_placeholder = st.empty() # Pour afficher le tracé


# --- Logique d'Optimisation (déclenchée par le bouton) ---
if start_button:
    # Réinitialiser l'état précédent
    st.session_state.optimization_results = None
    st.session_state.log_messages = ["Initialisation..."]
    st.session_state.final_plot_data = None
    st.session_state.result_df = None
    log_placeholder.text_area("Logs", "".join(st.session_state.log_messages), height=200, key="log_area_running")
    plot_placeholder.empty() # Effacer l'ancien tracé
    results_placeholder.empty() # Effacer les anciens résultats


    # Validation des paramètres avant de lancer
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
        # Copier les paramètres pour cette exécution
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
        st.session_state.last_params = params_run # Sauvegarder pour référence

        # Préparation pour l'optimisation
        optimizer_args = (params_run['nSub'], params_run['l0'], params_run['mse_target_angles_rad'], params_run['num_layers'], params_run['angle_weights'])
        iteration_count_st = 0 # Compteur spécifique à Streamlit

        # --- Callback pour dual_annealing (adapté pour Streamlit) ---
        def optimization_callback_st(xk, f, context):
            nonlocal iteration_count_st
            iteration_count_st += 1

            # Loguer le statut moins fréquemment pour ne pas surcharger
            if iteration_count_st % 10 == 0:
                 phase_str = ""
                 if context == 1: phase_str = "(Annealing)"
                 elif context == 2: phase_str = "(Local Search)"
                 log_msg = f"Callback {iteration_count_st} {phase_str}... MSE={f:.3e}"
                 st.session_state.log_messages.append(log_msg + "\n")
                 # Note: On ne peut pas mettre à jour log_placeholder directement ici
                 # car la fonction s'exécute de manière bloquante. Les logs
                 # seront affichés après la fin.

            # Pas de return True/False pour arrêter dual_annealing via callback

        # --- Exécution de l'Optimisation ---
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
                # Pré-compilation Numba (si pas déjà fait)
                # st.write("Vérification/Précompilation Numba...")
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


            # --- Traitement des Résultats ---
            st.session_state.optimization_results = result # Stocker l'objet résultat
            final_status = "Erreur" if error_msg else ("Succès" if result and result.success else "Échec/Interrompu")
            final_message = error_msg if error_msg else (result.message if result else "N/A")

            results_dict = {
                "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Algorithm": "Dual Annealing",
                "Status": final_status,
                "Message": final_message,
                "Num_Layers": params_run.get('num_layers', 'N/A'),
                "a_Substrat": params_run.get('a_substrat', 'N/A'),
                "nSub_real": params_run.get('nSub', np.nan+0j).real,
                "nSub_imag": params_run.get('nSub', np.nan+0j).imag,
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
                "nH_n": np.nan, "nH_k": np.nan,
                "nL_n": np.nan, "nL_k": np.nan,
                "Epaisseur_um": np.nan,
                "Epsilon_h_real": np.nan, "Epsilon_h_imag": np.nan,
                "Epsilon_l_real": np.nan, "Epsilon_l_imag": np.nan
            }

            best_params_tuple = None
            best_mse = np.inf

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
                    "nH_n": f"{final_best_nH.real:.6f}", "nH_k": f"{final_best_nH.imag:.6f}",
                    "nL_n": f"{final_best_nL.real:.6f}", "nL_k": f"{final_best_nL.imag:.6f}",
                    "Epaisseur_um": f"{final_best_epaisseur:.6f}",
                    "Epsilon_h_real": f"{epsilon_h.real:.6f}", "Epsilon_h_imag": f"{epsilon_h.imag:.6f}",
                    "Epsilon_l_real": f"{epsilon_l.real:.6f}", "Epsilon_l_imag": f"{epsilon_l.imag:.6f}"
                 })

                 st.session_state.log_messages.append(f"  nH = {format_complex(final_best_nH)}\n")
                 st.session_state.log_messages.append(f"  nL = {format_complex(final_best_nL)}\n")
                 st.session_state.log_messages.append(f"  Épaisseur = {final_best_epaisseur:.6f} µm\n")

                 # Préparer les données pour le tracé final
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
                    st.session_state.final_plot_data = None # Assurer qu'on ne trace pas si erreur

            else:
                 # Cas où l'optimisation échoue sans retourner de 'x' ou 'fun' valides
                 st.session_state.log_messages.append(f"Statut final: {final_status} - Aucun résultat utilisable trouvé.\n")
                 results_dict["Best_MSE_Weighted"] = "N/A"


            # Mettre à jour le statut final
            if final_status == "Succès":
                 status.update(label=f"Optimisation Réussie! MSE={best_mse:.3e}", state="complete", expanded=False)
            else:
                 status.update(label=f"Optimisation Terminée ({final_status})", state="error" if final_status=="Erreur" else "complete", expanded=True)

            # Stocker le DataFrame des résultats pour affichage et téléchargement
            st.session_state.result_df = pd.DataFrame([results_dict])

    else:
        st.error("Veuillez corriger les erreurs dans les paramètres avant de lancer l'optimisation.")
        # Assurer que l'état est propre si on clique sur start avec des erreurs
        st.session_state.optimization_results = None
        st.session_state.log_messages = ["Erreur de paramètres détectée."]
        st.session_state.final_plot_data = None
        st.session_state.result_df = None


# --- Affichage des Résultats (s'exécute après le bloc du bouton si des résultats existent) ---

# Afficher les logs collectés
log_placeholder.text_area("Logs", "".join(st.session_state.log_messages), height=200, key="log_area_final", disabled=True)

# Afficher les résultats numériques finaux
if st.session_state.result_df is not None:
    with results_placeholder.container():
        st.subheader("Résumé des Résultats")
        df_display = st.session_state.result_df.iloc[0] # Prendre la première ligne (il n'y en a qu'une)
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Statut", df_display.get("Status", "N/A"))
            st.metric("Meilleur MSE Pondéré", df_display.get("Best_MSE_Weighted", "N/A"))
            st.metric("nH (n+ik)", f"{df_display.get('nH_n','?')}+{df_display.get('nH_k','?')}j")
            st.metric("nL (n+ik)", f"{df_display.get('nL_n','?')}+{df_display.get('nL_k','?')}j")
            st.metric("Épaisseur (µm)", df_display.get("Epaisseur_um", "N/A"))
        with res_col2:
            st.metric("Durée (s)", df_display.get("Opt_Duration_s", "N/A"))
            st.metric("NFEV (Évals Func.)", df_display.get("NFEV", "N/A"))
            st.metric("NIT (Itér. Globales)", df_display.get("NIT_Global", "N/A"))
            st.metric("Couches (N)", df_display.get("Num_Layers", "N/A"))
            st.metric("λ (µm)", df_display.get("Lambda_l0", "N/A"))

        with st.expander("Voir tous les détails des paramètres et résultats"):
            st.dataframe(st.session_state.result_df.T) # Transposer pour meilleure lisibilité

        # Bouton de téléchargement Excel
        excel_data = convert_df_to_excel(st.session_state.result_df)
        st.download_button(
            label="📥 Télécharger les résultats (Excel)",
            data=excel_data,
            file_name=f"optimization_results_SA_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# Afficher le tracé final
if st.session_state.final_plot_data is not None:
    with plot_placeholder.container():
        plot_data = st.session_state.final_plot_data
        fig, ax = plt.subplots(figsize=(7, 5))

        valid_s = ~np.isnan(plot_data["angles_deg"]) & ~np.isnan(plot_data["Rs"])
        valid_p = ~np.isnan(plot_data["angles_deg"]) & ~np.isnan(plot_data["Rp"])

        ax.plot(plot_data["angles_deg"][valid_s], plot_data["Rs"][valid_s], label='Rs', linestyle='-', color='blue', marker='.', markersize=3)
        ax.plot(plot_data["angles_deg"][valid_p], plot_data["Rp"][valid_p], label='Rp', linestyle='--', color='red', marker='x', markersize=3)

        # Marquer la zone cible MSE
        if 'last_params' in st.session_state and st.session_state.last_params:
             mse_angles = st.session_state.last_params['mse_target_angles_deg']
             if len(mse_angles) > 0:
                  ax.axvspan(mse_angles[0], mse_angles[-1], color='lightgray', alpha=0.3, label='Zone Cible MSE')


        ax.set_xlabel("Angle d'incidence (degrés)")
        ax.set_ylabel("Réflectance (échelle log)")
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize='small')
        ax.set_ylim(1e-6, 1.1)
        ax.set_xlim(0, 90)

        nH_str = format_complex(plot_data['nH'])
        nL_str = format_complex(plot_data['nL'])
        nSub_str = format_complex(plot_data['nSub'])
        ep_str = f"{plot_data['ep']:.4f}"
        mse_str = f"{plot_data['mse']:.3e}"
        l0_str = f"{plot_data['l0']:.2f}"
        status_str = plot_data['status']

        title = (f"Résultat Final ({status_str}) | MSE_pondéré: {mse_str} | λ={l0_str}µm\n"
                 f"nH={nH_str}, nL={nL_str}, ép={ep_str}µm | nSub={nSub_str}, N={plot_data['N']}")
        ax.set_title(title, fontsize=9)

        fig.tight_layout()
        st.pyplot(fig)
else:
     # Afficher un message si aucun tracé n'est disponible (par ex. après erreur ou avant la 1ère exécution)
     with plot_placeholder.container():
          st.info("Le tracé de réflectance apparaîtra ici après une optimisation réussie.")

# --- Pied de page ou informations supplémentaires ---
st.sidebar.divider()
st.sidebar.caption(f"Date/Heure actuelle: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
