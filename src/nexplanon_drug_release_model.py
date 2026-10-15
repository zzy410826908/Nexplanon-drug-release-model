"""
Nexplanon drug-release fitting workflow.

This script is the single entry point for:
1. fitting DOS, PDL, ha_skin, and ha_ends to Organon data,
2. exporting the full Paul-model simulation table, and
3. generating diagnostic/result plots.

It supersedes the older split workflow where one script generated
`Paul_model_results.csv` and another script plotted it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import textwrap
from scipy.optimize import fsolve, brentq, minimize
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

# =====================================================
# MEASURED PARAMETERS (not fitted)
# =====================================================
L      = 4.0
R_o    = 0.100
R_i    = 0.094
V_core = np.pi * R_i**2 * L
SA_ends = 2 * np.pi * R_i**2

Cs_28  = 2.30e-03;  D_28  = 3.30e-09
Cs_DL  = 1.92e-03;  D_DL  = 6.36e-09
Cs_15  = 1.30e-03;  D_15  = 2.20e-09
Cs_a   = 6.90e-06;  D_a   = 7.02e-06
# Measured PDL is retained only as the reference scale for optimisation.
# PDL itself is fitted to the release data below.
P_DL   = Cs_DL * D_DL
P_15   = Cs_15 * D_15
P_a    = Cs_a  * D_a

Load           = 68                      # mg
C0             = Load / 1000 / V_core    # g/cm³
# Corrected from the 100 mL, daily-medium-replacement calculation: the
# time-averaged medium concentration is 2.30% of water solubility.
sink_condition = 0.0230
C_media        = sink_condition * Cs_a
zero_order_ends = 0                   # mcg/day

t_days_base_early = np.arange(0, 61, 1)
t_days_burst      = np.arange(0, 31, 1)  # burst evaluated for first 30 days only

INITIAL_GUESS = [1.4, P_DL, 0.128, 0.0008]
FIT_BOUNDS = [(1.0, 5.0), (0.05 * P_DL, 20.0 * P_DL), (1e-4, 5.0), (1e-4, 0.1)]
DATA_DURATION_CASES = {
    "3 months": 90,
    "6 months": 180,
    "1 year": 365,
    "2 years": 730,
    "3 years": 1095,
}

# =====================================================
# SIMULATION FUNCTION
# =====================================================

def simulate(DOS, PDL, ha_skin, ha_ends, P15_factor=None, Ri_override=None,
             zero_order_ends_override=None, verbose=False, return_details=False):
    """
    Run the full 4-part Paul model simulation for given fitted parameters.

    Parameters
    ----------
    DOS     : degree of supersaturation (> 1)
    PDL     : fitted permeability of the depletion layer (g/(cm·s))
    ha_skin : skin water-layer thickness (cm)
    ha_ends : ends water-layer thickness (cm)
    Ri_override : optional inner radius (cm); R_o remains fixed

    Returns
    -------
    if return_details is False:
        t_days : np.ndarray   – simulation time points (days)
        m_total: np.ndarray   – m_total_effective (mcg/day)
    else:
        dict of arrays with full model outputs
    """
    Ri_local = R_i if Ri_override is None else float(Ri_override)
    if Ri_local <= 0 or Ri_local >= R_o:
        raise ValueError("Ri_override must be positive and smaller than R_o.")
    V_core_local = np.pi * Ri_local**2 * L
    SA_ends_local = 2 * np.pi * Ri_local**2
    C0_local = Load / 1000 / V_core_local
    Ra_skin   = R_o + ha_skin
    C_core_t0 = DOS * Cs_28
    # allow temporary override of P_15 via multiplicative factor
    P15_local = P_15 if P15_factor is None else P_15 * float(P15_factor)
    # allow temporary override of zero-order ends baseline
    zero_order_local = zero_order_ends if zero_order_ends_override is None else float(zero_order_ends_override)

    # ── PART 1: Burst ────────────────────────────────────────────────────────
    C1_burst = 2.0 * np.pi * L / (
        np.log(R_o / Ri_local) / P15_local +
        np.log(Ra_skin / R_o) / P_a
    )
    C2_burst = 1.0 - C_media / Cs_a
    C3_burst = C_core_t0 / Cs_28 - 1.0

    t_sec_burst    = t_days_burst * 86400
    mskin_burst_gs = C1_burst * (C2_burst + C3_burst *
                     np.exp(-C1_burst / (Cs_28 * V_core_local) * t_sec_burst))
    mskin_burst_mcgday = mskin_burst_gs * 1e6 * 86400

    # ── PART 2a: Rd(t) solver ────────────────────────────────────────────────
    C1_rd = (-1.0) * (1.0 - sink_condition) / (C0_local - Cs_DL)
    C2_rd = 1.0 / PDL
    C4_rd = np.log(R_o / Ri_local) / P15_local + np.log(Ra_skin / R_o) / P_a
    A_rd  = C2_rd * (2.0 * np.log(Ri_local) + 1.0) + 2.0 * C4_rd

    def t_sec_from_rd(Rd):
        return (
            (Rd**2 - Ri_local**2) * A_rd
            + 2.0 * C2_rd * (-Rd**2 * np.log(Rd) + Ri_local**2 * np.log(Ri_local))
        ) / (4.0 * C1_rd)

    Rd_min_bound    = 1e-12
    t_sec_max       = t_sec_from_rd(Rd_min_bound)
    # Add 13 radius-spaced depletion-layer points from 5% to 0.001% of R_i.
    # Logarithmic spacing provides useful resolution across the 5,000-fold range.
    Rd_radius_points = np.geomspace(Ri_local * 0.05, Ri_local * 0.00001, 13)
    Rd_threshold    = Rd_radius_points[-1]
    t_day_threshold = t_sec_from_rd(Rd_threshold) / 86400

    t_days_late_rd = np.arange(90, np.ceil(t_day_threshold) + 30, 30)
    t_days_radius_points = np.array([
        t_sec_from_rd(Rd) / 86400 for Rd in Rd_radius_points
    ])
    t_days_rd_grid = np.unique(
        np.concatenate([
            t_days_base_early, t_days_late_rd, t_days_radius_points,
            [t_day_threshold]
        ])
    )
    t_days_rd_grid = t_days_rd_grid[t_days_rd_grid <= t_day_threshold + 1e-12]

    Rd_solutions = []
    t_days_out   = []
    t_secs_out   = []

    for t_day in t_days_rd_grid:
        t_sec = t_day * 86400
        if t_sec > t_sec_max:
            break
        if t_sec == 0.0:
            Rd_val = Ri_local
        else:
            try:
                Rd_val = brentq(lambda Rd: t_sec_from_rd(Rd) - t_sec,
                                Rd_min_bound, Ri_local)
            except Exception:
                Rd_val = np.nan
        if not np.isnan(Rd_val) and (Rd_val <= 0 or Rd_val > Ri_local):
            Rd_val = np.nan
        Rd_solutions.append(Rd_val)
        t_days_out.append(t_day)
        t_secs_out.append(t_sec)
        if not np.isnan(Rd_val) and Rd_val < Rd_threshold:
            break

    t_days_sim    = np.array(t_days_out)
    t_seconds_sim = np.array(t_secs_out)

    # ── PART 2b: m_skin_sustained ────────────────────────────────────────────
    def _mskin_sus(Rd):
        if Rd <= 0 or np.isnan(Rd):
            return np.nan
        return (2.0 * np.pi * L * (1.0 - sink_condition)) / (
            np.log(Ri_local / Rd) / PDL +
            np.log(R_o / Ri_local) / P15_local +
            np.log(Ra_skin / R_o) / P_a
        )

    mskin_sus_mcgday = np.array([_mskin_sus(Rd) for Rd in Rd_solutions]) * 1e6 * 86400

    # ── PART 3: m_ends ───────────────────────────────────────────────────────
    C1_ends = (-1.0) * (1.0 - sink_condition) / (C0_local - Cs_DL)
    C2_ends = (L / (2.0 * PDL)) + ha_ends / P_a
    C3_ends = -1.0 / PDL

    def _zd_res(Zd, t_sec):
        return (
            (0.5 * C3_ends * Zd**2 + C2_ends * Zd
             - C2_ends * L / 2.0 - C3_ends * L**2 / 8.0) / C1_ends - t_sec
        )

    Zd_solutions = []
    Zd_guess = L / 2.0
    for t_sec in t_seconds_sim:
        if t_sec == 0.0:
            Zd_solutions.append(L / 2.0)
        else:
            try:
                sol = fsolve(_zd_res, Zd_guess, args=(t_sec,), full_output=True)
                Zd_val = float(sol[0][0])
                if Zd_val <= 0 or Zd_val >= L / 2.0:
                    Zd_val = np.nan
                else:
                    Zd_guess = Zd_val
            except Exception:
                Zd_val = np.nan
            Zd_solutions.append(Zd_val)

    def _mends(Zd):
        if np.isnan(Zd) or Zd > L / 2.0:
            return np.nan
        denom = (L / 2.0 - Zd) / PDL + ha_ends / P_a
        if denom <= 0:
            return np.nan
        return (1.0 - sink_condition) * SA_ends_local / denom

    mends_mcgday = np.array([_mends(Zd) for Zd in Zd_solutions])
    mends_mcgday = np.where(np.isnan(mends_mcgday), np.nan,
                            mends_mcgday * 1e6 * 86400 + zero_order_local)

    # ── PART 4: Combine ──────────────────────────────────────────────────────
    def cum_mcg(times, rates):
        Q = [0.0]
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            r0, r1 = rates[i - 1], rates[i]
            if np.isnan(r0) or np.isnan(r1):
                Q.append(Q[-1])
            elif i == 1:
                Q.append(Q[-1] + dt * r1)
            else:
                Q.append(Q[-1] + dt * (r0 + r1) / 2.0)
        return np.array(Q)

    burst_interp = np.interp(t_days_sim, t_days_burst, mskin_burst_mcgday,
                             left=mskin_burst_mcgday[0], right=0.0)

    # Net burst = excess above sustained release, but it is forced to decay to
    # zero exactly at day 30 so the combined skin-release curve transitions
    # smoothly to the sustained-only regime.
    raw_burst_extra = np.where(
        (t_days_sim <= 30) & (~np.isnan(mskin_sus_mcgday)),
        np.maximum(burst_interp - mskin_sus_mcgday, 0.0), 0.0
    )
    burst_extra_day30 = raw_burst_extra[np.where(np.isclose(t_days_sim, 30))[0][0]] if np.any(np.isclose(t_days_sim, 30)) else 0.0
    taper = np.where(t_days_sim <= 30, t_days_sim / 30.0, 1.0)
    burst_extra = np.where(
        t_days_sim <= 30,
        np.maximum(raw_burst_extra - burst_extra_day30 * taper, 0.0),
        0.0
    )

    mskin_comb = np.where(np.isnan(mskin_sus_mcgday), np.nan,
                          mskin_sus_mcgday + burst_extra)

    Qskin_mg = cum_mcg(t_days_sim, mskin_comb) / 1000.0
    Qends_mg = cum_mcg(t_days_sim, mends_mcgday) / 1000.0

    mskin_eff = np.array([
        mskin_comb[i] * max(1.0 - Qends_mg[i] / Load, 0.0)
        if not np.isnan(mskin_comb[i]) else np.nan
        for i in range(len(t_days_sim))
    ])
    mends_eff = np.array([
        mends_mcgday[i] * max(1.0 - Qskin_mg[i] / Load, 0.0)
        if not np.isnan(mends_mcgday[i]) else np.nan
        for i in range(len(t_days_sim))
    ])
    m_total = np.where(
        np.isnan(mskin_eff) | np.isnan(mends_eff), np.nan,
        mskin_eff + mends_eff
    )
    # Cumulative total release is used to report drug remaining in the implant.
    Q_skin_effective_mg = cum_mcg(t_days_sim, mskin_eff) / 1000.0
    Q_ends_effective_mg = cum_mcg(t_days_sim, mends_eff) / 1000.0
    Q_total_effective_mg = cum_mcg(t_days_sim, m_total) / 1000.0

    if return_details:
        return {
            "t_days": t_days_sim,
            "Rd": np.array(Rd_solutions),
            "Zd": np.array(Zd_solutions),
            "m_skin_burst": burst_interp,
            "m_skin_burst_net": burst_extra,
            "m_skin_sustained": mskin_sus_mcgday,
            "m_skin_combined": mskin_comb,
            "m_skin_effective": mskin_eff,
            "m_ends": mends_mcgday,
            "m_ends_effective": mends_eff,
            "m_total": m_total,
            "Q_skin_effective_mg": Q_skin_effective_mg,
            "Q_ends_effective_mg": Q_ends_effective_mg,
            "Q_total_effective_mg": Q_total_effective_mg,
            "Q_skin_mg": Qskin_mg,
            "Q_ends_mg": Qends_mg,
        }

    return t_days_sim, m_total


def objective_for_dataset(params, t_exp_fit, m_exp_fit):
    """Sum of squared residuals for a chosen experimental dataset."""
    DOS, PDL, ha_skin, ha_ends = params

    if any(value < lower or value > upper for value, (lower, upper) in zip(params, FIT_BOUNDS)):
        return 1e12

    try:
        t_sim, m_sim = simulate(DOS, PDL, ha_skin, ha_ends)
    except Exception:
        return 1e12

    valid = ~np.isnan(m_sim)
    if valid.sum() < 3:
        return 1e12

    m_sim_interp = np.interp(t_exp_fit, t_sim[valid], m_sim[valid])
    return np.sum((m_sim_interp - m_exp_fit) ** 2)


def fit_dataset(t_exp_fit, m_exp_fit, x0=None, maxiter=2000, disp=False):
    """Fit DOS, PDL, ha_skin, and ha_ends to a provided experimental dataset."""
    if x0 is None:
        x0 = INITIAL_GUESS

    # PDL is ~1e-11 while the other fit parameters are order 1 to 1e-4.
    # Optimise its ratio to the measured reference value, then return PDL in
    # physical units so all downstream reporting remains interpretable.
    def normalise(params):
        DOS, PDL, ha_skin, ha_ends = params
        return np.array([DOS, PDL / P_DL, ha_skin, ha_ends], dtype=float)

    def denormalise(params):
        DOS, PDL_ratio, ha_skin, ha_ends = params
        return np.array([DOS, PDL_ratio * P_DL, ha_skin, ha_ends], dtype=float)

    result = minimize(
        lambda scaled_params: objective_for_dataset(denormalise(scaled_params), t_exp_fit, m_exp_fit),
        normalise(x0),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": maxiter, "disp": disp}
    )
    result.x = denormalise(result.x)
    return result


def calculate_fit_metrics(t_exp_eval, m_exp_eval, params):
    """Evaluate a fitted parameter set against any experimental dataset."""
    DOS, PDL, ha_skin, ha_ends = params
    t_sim, m_sim = simulate(DOS, PDL, ha_skin, ha_ends)
    valid = ~np.isnan(m_sim)
    m_sim_interp = np.interp(t_exp_eval, t_sim[valid], m_sim[valid])
    residuals = m_sim_interp - m_exp_eval

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / m_exp_eval)) * 100
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((m_exp_eval - np.mean(m_exp_eval))**2)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "t_sim": t_sim,
        "m_sim": m_sim,
        "m_sim_interp": m_sim_interp,
        "residuals": residuals,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }


def run_data_duration_sensitivity(base_dir, t_exp_full, m_exp_full, m_exp_sd_full=None, pdf=None):
    """Run the former stand-alone duration sensitivity study from this file."""
    print(f"\n{'=' * 80}")
    print("DATA DURATION SENSITIVITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"\nFull dataset: {len(t_exp_full)} points, spanning {t_exp_full[-1]:.1f} days (~{t_exp_full[-1]/365:.2f} years)")

    results = {}

    for duration_name, duration_days in DATA_DURATION_CASES.items():
        print(f"\n{'─' * 80}")
        print(f"FITTING WITH {duration_name.upper()} OF DATA ({duration_days} days)")
        print(f"{'─' * 80}")

        mask = t_exp_full <= duration_days
        t_exp_trunc = t_exp_full[mask]
        m_exp_trunc = m_exp_full[mask]

        print(f"Data points: {len(t_exp_trunc)} (from {len(t_exp_full)} total)")
        print(f"Time range: {t_exp_trunc[0]:.1f} - {t_exp_trunc[-1]:.1f} days")
        print("  Optimizing...")

        result = fit_dataset(t_exp_trunc, m_exp_trunc, x0=INITIAL_GUESS, maxiter=5000, disp=False)
        params = tuple(result.x)

        trunc_metrics = calculate_fit_metrics(t_exp_trunc, m_exp_trunc, params)
        full_metrics = calculate_fit_metrics(t_exp_full, m_exp_full, params)

        print(f"\nFitted parameters:")
        print(f"  DOS       = {params[0]:.6f}")
        print(f"  PDL       = {params[1]:.6e} g/(cm·s)")
        print(f"  ha_skin   = {params[2]:.8f} cm")
        print(f"  ha_ends   = {params[3]:.8f} cm")
        print(f"  Converged: {result.success}")
        print(f"  Objective value: {result.fun:.3f}")

        print(f"\nPrediction errors (on {duration_name} data):")
        print(f"  MAE   = {trunc_metrics['MAE']:.3f} mcg/day")
        print(f"  RMSE  = {trunc_metrics['RMSE']:.3f} mcg/day")
        print(f"  MAPE  = {trunc_metrics['MAPE']:.2f} %")
        print(f"  R²    = {trunc_metrics['R2']:.4f}")

        print(f"\nPrediction errors (on FULL dataset):")
        print(f"  MAE   = {full_metrics['MAE']:.3f} mcg/day")
        print(f"  RMSE  = {full_metrics['RMSE']:.3f} mcg/day")
        print(f"  MAPE  = {full_metrics['MAPE']:.2f} %")
        print(f"  R²    = {full_metrics['R2']:.4f}")

        results[duration_name] = {
            "params": params,
            "n_points": len(t_exp_trunc),
            "duration_days": duration_days,
            "result": result,
            "truncated": trunc_metrics,
            "full": full_metrics,
        }

    durations_list = list(results.keys())

    print(f"\n{'=' * 80}")
    print("SUMMARY: FITTING ACCURACY VS DATA DURATION")
    print(f"{'=' * 80}\n")

    summary_df = pd.DataFrame({
        "Data Duration": durations_list,
        "# Points": [results[d]["n_points"] for d in durations_list],
        "MAE (full)": [results[d]["full"]["MAE"] for d in durations_list],
        "RMSE (full)": [results[d]["full"]["RMSE"] for d in durations_list],
        "MAPE % (full)": [results[d]["full"]["MAPE"] for d in durations_list],
        "R² (full)": [results[d]["full"]["R2"] for d in durations_list],
    })
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary_df.to_csv(base_dir / "data_duration_sensitivity_summary.csv", index=False)

    print(f"\n{'=' * 80}")
    print("FITTED PARAMETERS ACROSS DATA DURATIONS")
    print(f"{'=' * 80}\n")

    params_df = pd.DataFrame({
        "Data Duration": durations_list,
        "DOS": [results[d]["params"][0] for d in durations_list],
        "PDL (g/(cm·s))": [results[d]["params"][1] for d in durations_list],
        "ha_skin (cm)": [results[d]["params"][2] for d in durations_list],
        "ha_ends (cm)": [results[d]["params"][3] for d in durations_list],
    })
    print(params_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    params_df.to_csv(base_dir / "data_duration_fitted_parameters.csv", index=False)

    fig_summary = plt.figure(figsize=(14, 10))
    fig_summary.suptitle("Figure 3. Data-duration sensitivity summary", fontsize=15, y=0.98)

    ax_top = fig_summary.add_axes([0.05, 0.55, 0.90, 0.35])
    ax_top.axis("off")
    ax_top.text(0.0, 1.05, "Table 3. Accuracy on full dataset", fontsize=12, fontweight="bold", transform=ax_top.transAxes)
    summary_rows = [
        [
            row["Data Duration"],
            f"{int(row['# Points'])}",
            f"{row['MAE (full)']:.4f}",
            f"{row['RMSE (full)']:.4f}",
            f"{row['MAPE % (full)']:.4f}",
            f"{row['R² (full)']:.4f}",
        ]
        for _, row in summary_df.iterrows()
    ]
    tbl_summary = ax_top.table(
        cellText=summary_rows,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="upper left",
        bbox=[0, 0.02, 1.0, 0.88],
    )
    tbl_summary.auto_set_font_size(False)
    tbl_summary.set_fontsize(9)
    tbl_summary.scale(1, 1.2)

    ax_bottom = fig_summary.add_axes([0.05, 0.08, 0.90, 0.35])
    ax_bottom.axis("off")
    ax_bottom.text(0.0, 1.05, "Table 4. Fitted parameters across data durations", fontsize=12, fontweight="bold", transform=ax_bottom.transAxes)
    tbl_params = ax_bottom.table(
        cellText=[[
            row[0], f"{row[1]:.6f}", f"{row[2]:.6e}",
            f"{row[3]:.6f}", f"{row[4]:.6f}"
        ] for row in params_df.values],
        colLabels=params_df.columns,
        cellLoc="center",
        loc="upper left",
        bbox=[0, 0.02, 1.0, 0.88],
    )
    tbl_params.auto_set_font_size(False)
    tbl_params.set_fontsize(9)
    tbl_params.scale(1, 1.2)
    if pdf is not None:
        pdf.savefig(fig_summary)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle('Figure 4. Duration-sensitivity metrics', fontsize=14, y=0.99)

    ax = axes[0]
    mae_full = [results[d]["full"]["MAE"] for d in durations_list]
    ax.plot(range(len(durations_list)), mae_full, 'o-', linewidth=2.5, markersize=10, color='#d62728')
    ax.set_xticks(range(len(durations_list)))
    ax.set_xticklabels(durations_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('MAE (mcg/day)', fontsize=11, fontweight='bold')
    ax.set_title('A. Mean Absolute Error vs Data Duration', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    r2_full = [results[d]["full"]["R2"] for d in durations_list]
    ax.plot(range(len(durations_list)), r2_full, 'o-', linewidth=2.5, markersize=10, color='#1f77b4')
    ax.set_xticks(range(len(durations_list)))
    ax.set_xticklabels(durations_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('R² (full dataset)', fontsize=11, fontweight='bold')
    ax.set_title('B. Model Fit Quality vs Data Duration', fontsize=12, fontweight='bold')
    # Include the lowest duration-specific fit, including the 3-month result.
    ax.set_ylim([max(0.0, min(r2_full) - 0.01), 1.0])
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    mape_full = [results[d]["full"]["MAPE"] for d in durations_list]
    ax.plot(range(len(durations_list)), mape_full, 'o-', linewidth=2.5, markersize=10, color='#ff7f0e')
    ax.set_xticks(range(len(durations_list)))
    ax.set_xticklabels(durations_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    ax.set_title('C. Mean Absolute Percentage Error\nvs Data Duration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)

    fig5, ax = plt.subplots(1, 1, figsize=(10, 6))
    selected_durations = ["6 months", "1 year", "2 years", "3 years"]
    colors_cycle = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    has_sd = m_exp_sd_full is not None and np.any(~np.isnan(m_exp_sd_full))
    if has_sd:
        valid_sd = ~np.isnan(m_exp_sd_full)
        ax.errorbar(
            t_exp_full[valid_sd] / 365, m_exp_full[valid_sd], yerr=m_exp_sd_full[valid_sd],
            fmt='o', color='black', markersize=6, capsize=4, linewidth=1.2,
            label='Experimental data (±SD)', zorder=3, ecolor='black', alpha=0.85
        )
        ax.scatter(
            t_exp_full[~valid_sd] / 365, m_exp_full[~valid_sd],
            s=40, alpha=0.75, color='black', zorder=3, edgecolors='black', linewidth=0.5
        )
    else:
        ax.scatter(
            t_exp_full / 365, m_exp_full,
            s=55, alpha=0.75, color='black', label='Experimental data',
            zorder=3, edgecolors='black', linewidth=0.5
        )

    for duration_name, color in zip(selected_durations, colors_cycle):
        fit_data = results[duration_name]
        sim_metrics = fit_data["full"]
        t_sim = sim_metrics["t_sim"]
        m_sim = sim_metrics["m_sim"]

        mask_sim_nonzero = t_sim > 0
        t_sim_plot = t_sim[mask_sim_nonzero]
        m_sim_plot = m_sim[mask_sim_nonzero]
        valid = ~np.isnan(m_sim_plot)

        PDL_fit = fit_data["params"][1]
        ax.plot(
            t_sim_plot[valid] / 365,
            m_sim_plot[valid],
            linewidth=1.3,
            color=color,
            label=(
                f'{duration_name}: PDL={PDL_fit:.3e} g/(cm·s), '
                f'R²={fit_data["full"]["R2"]:.4f}'
            ),
            zorder=4,
        )

    ax.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Release Rate (mcg/day)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 6. Drug release profiles using PDL fitted from 6, 12, 24, and 36 months data', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    profile_max_days = max(
        np.nanmax(results[duration_name]["full"]["t_sim"])
        for duration_name in selected_durations
    )
    profile_max_years = int(np.ceil(profile_max_days / 365.0))
    ax.set_xlim([-0.1, profile_max_years])
    plt.tight_layout()
    # Save Figure 6 after Figure 5 so it follows Figure 4 and Figure 5 in the report.

    # Figure 7: contrast a short-term fit with progressively longer fits.
    fig7, ax7 = plt.subplots(1, 1, figsize=(10, 6))
    selected_durations_fig7 = ["3 months", "6 months", "1 year", "3 years"]
    colors_fig7 = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    if has_sd:
        valid_sd = ~np.isnan(m_exp_sd_full)
        ax7.errorbar(
            t_exp_full[valid_sd] / 365.0, m_exp_full[valid_sd], yerr=m_exp_sd_full[valid_sd],
            fmt='o', color='black', markersize=6, capsize=4, linewidth=1.2,
            label='Experimental data (±SD)', zorder=3, ecolor='black', alpha=0.85
        )
        ax7.scatter(
            t_exp_full[~valid_sd] / 365.0, m_exp_full[~valid_sd],
            s=40, alpha=0.75, color='black', zorder=3, edgecolors='black', linewidth=0.5
        )
    else:
        ax7.scatter(
            t_exp_full / 365.0, m_exp_full,
            s=55, alpha=0.75, color='black', label='Experimental data',
            zorder=3, edgecolors='black', linewidth=0.5
        )

    for duration_name, color in zip(selected_durations_fig7, colors_fig7):
        fit_data = results[duration_name]
        t_sim = fit_data["full"]["t_sim"]
        m_sim = fit_data["full"]["m_sim"]
        valid = (t_sim > 0) & ~np.isnan(m_sim)
        PDL_fit = fit_data["params"][1]
        ha_skin_fit = fit_data["params"][2]
        ax7.plot(
            t_sim[valid] / 365.0, m_sim[valid], linewidth=1.5, color=color,
            label=(
                f'{duration_name}: PDL={PDL_fit:.1e} g/(cm·s), ha_skin={ha_skin_fit:.3f} cm, '
                f'R²={fit_data["full"]["R2"]:.3f}'
            ), zorder=4,
        )

    ax7.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Release Rate (mcg/day)', fontsize=11, fontweight='bold')
    ax7.set_title(
        'Figure 7. Drug release profiles using parameters fitted from\n'
        '3 months, 6 months, 1 year, and 3 years data',
        fontsize=12, fontweight='bold'
    )
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=9, loc='upper right')
    profile_max_days_fig7 = max(
        np.nanmax(results[duration_name]["full"]["t_sim"])
        for duration_name in selected_durations_fig7
    )
    profile_max_years_fig7 = int(np.ceil(profile_max_days_fig7 / 365.0))
    ax7.set_xlim([0, profile_max_years_fig7])
    ax7.set_xticks(np.arange(0, profile_max_years_fig7 + 1, 1))
    ax7.tick_params(axis='x', labelsize=9)
    plt.tight_layout()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    fig.suptitle('Figure 5. Parameter convergence vs data duration', fontsize=13, y=0.99)
    dos_values = [results[d]["params"][0] for d in durations_list]
    PDL_values = [results[d]["params"][1] for d in durations_list]
    ha_skin_values = [results[d]["params"][2] for d in durations_list]
    ha_ends_values = [results[d]["params"][3] for d in durations_list]

    ax = axes[0]
    ax.plot(range(len(durations_list)), dos_values, 'o-', linewidth=2.5, markersize=10, color='#1f77b4')
    ax.set_xticks(range(len(durations_list)))
    ax.set_xticklabels(durations_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('DOS', fontsize=11, fontweight='bold')
    ax.set_title('A. DOS Convergence vs Data Duration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(len(durations_list)), PDL_values, 'o-', linewidth=2.5, markersize=10, color='#ff7f0e')
    ax.set_xticks(range(len(durations_list)))
    ax.set_xticklabels(durations_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('PDL (g/(cm·s))', fontsize=11, fontweight='bold')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_title('B. PDL Convergence vs Data Duration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(range(len(durations_list)), [x * 10 for x in ha_skin_values], 'o-', linewidth=2.5, markersize=10, color='#2ca02c')
    ax.set_xticks(range(len(durations_list)))
    ax.set_xticklabels(durations_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('ha_skin (mm)', fontsize=11, fontweight='bold')
    ax.set_title('C. ha_skin Convergence vs Data Duration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(range(len(durations_list)), ha_ends_values, 'o-', linewidth=2.5, markersize=10, color='#d62728')
    ax.set_xticks(range(len(durations_list)))
    ax.set_xticklabels(durations_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('ha_ends (cm)', fontsize=11, fontweight='bold')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0),
                        useOffset=False, useMathText=True)
    ax.set_title('D. ha_ends Convergence vs Data Duration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    if pdf is not None:
        pdf.savefig(fig5)
    plt.close(fig5)
    if pdf is not None:
        pdf.savefig(fig7)
    plt.close(fig7)
    plt.close(fig_summary)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")
    print("KEY INSIGHTS:")
    r2_vals = [results[d]['full']['R2'] for d in durations_list]
    best_idx = np.argmax(r2_vals)
    print(f"  • Best fit achieved with: {durations_list[best_idx]}")
    print(f"  • Best R²: {max(r2_vals):.4f}")
    print("  • Minimum improvement threshold (90% of max R²):")
    for dur_name, r2 in zip(durations_list, r2_vals):
        if r2 >= 0.9 * max(r2_vals):
            print(f"      → {dur_name} ({results[dur_name]['n_points']} points)")
            break

    return results

# =====================================================
# LOAD EXPERIMENTAL DATA
# =====================================================
REPORT_TITLE = "Nexplanon drug release model"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
# The source code lives in src/; data, reports, and results are stored at the
# repository root for a portable public-project layout.
BASE = (SCRIPT_DIRECTORY.parent
        if SCRIPT_DIRECTORY.name in {"src", "python codes"}
        else SCRIPT_DIRECTORY)
REPORTS_DIRECTORY = BASE / "reports"
REPORTS_DIRECTORY.mkdir(exist_ok=True)
RESULTS_DIRECTORY = BASE / "results"
RESULTS_DIRECTORY.mkdir(exist_ok=True)
DATA_DIRECTORY = BASE / "data"
DEFAULT_DATA_PATH = DATA_DIRECTORY / "release_rate_data.csv"


def parse_arguments():
    """Return the optional experimental-data path supplied at the command line."""
    parser = argparse.ArgumentParser(
        description="Fit the PSS model to Nexplanon total-release data."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "CSV with t_day and total_mcg_day columns, or the included FDA-data "
            "schema using release_rate_mcg_day (default: data/release_rate_data.csv)."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create output files without opening plot windows.",
    )
    return parser.parse_args()


args = parse_arguments()
data_path = args.data_path.expanduser().resolve()
if not data_path.is_file():
    raise FileNotFoundError(f"Experimental data file was not found: {data_path}")

exp = pd.read_csv(data_path)
if "total_mcg_day" not in exp.columns and "release_rate_mcg_day" in exp.columns:
    exp = exp.rename(columns={"release_rate_mcg_day": "total_mcg_day"})
if "SD" not in exp.columns and "sd_mcg_day" in exp.columns:
    exp = exp.rename(columns={"sd_mcg_day": "SD"})

required_columns = {"t_day", "total_mcg_day"}
missing_columns = required_columns.difference(exp.columns)
if missing_columns:
    raise ValueError(
        "Input CSV is missing required column(s): "
        + ", ".join(sorted(missing_columns))
    )

exp["t_day"] = pd.to_numeric(exp["t_day"], errors="coerce")
exp["total_mcg_day"] = pd.to_numeric(exp["total_mcg_day"], errors="coerce")
exp = exp.dropna(subset=["t_day", "total_mcg_day"]).sort_values("t_day")

# Exclude t=0 data point (initialization artifact with anomalously high values)
mask_nonzero = exp["t_day"].values > 0
exp = exp[mask_nonzero].reset_index(drop=True)

t_exp  = exp["t_day"].values
m_exp  = exp["total_mcg_day"].values
m_exp_sd = pd.to_numeric(exp["SD"], errors="coerce").values if "SD" in exp.columns else None

if len(t_exp) < 3:
    raise ValueError("At least three nonzero experimental data points are required for fitting.")

print(f"Experimental data: {data_path}")
print(f"Fitting {len(t_exp)} points spanning {t_exp[0]:.1f} to {t_exp[-1]:.1f} days.")

# =====================================================
# OBJECTIVE FUNCTION  (sum of squared residuals)
# =====================================================

def objective(params):
    return objective_for_dataset(params, t_exp, m_exp)

# =====================================================
# OPTIMISATION  (Nelder-Mead, no gradient needed)
# =====================================================
print("Starting optimisation …")
result = fit_dataset(t_exp, m_exp, x0=INITIAL_GUESS, maxiter=2000, disp=True)

DOS_fit, PDL_fit, ha_skin_fit, ha_ends_fit = result.x
print(f"\n{'='*55}")
print("FITTED PARAMETERS")
print(f"  DOS      = {DOS_fit:.4f}")
print(f"  PDL      = {PDL_fit:.6e} g/(cm·s)")
print(f"  ha_skin  = {ha_skin_fit:.6f} cm")
print(f"  ha_ends  = {ha_ends_fit:.6f} cm")
print(f"  Converged: {result.success}  ({result.message})")

# =====================================================
# PREDICTION ERRORS
# =====================================================
t_sim, m_sim = simulate(DOS_fit, PDL_fit, ha_skin_fit, ha_ends_fit)

valid        = ~np.isnan(m_sim)
m_sim_interp = np.interp(t_exp, t_sim[valid], m_sim[valid])
residuals    = m_sim_interp - m_exp

MAE  = np.mean(np.abs(residuals))
RMSE = np.sqrt(np.mean(residuals**2))
MAPE = np.mean(np.abs(residuals / m_exp)) * 100
SS_res = np.sum(residuals**2)
SS_tot = np.sum((m_exp - np.mean(m_exp))**2)
R2   = 1.0 - SS_res / SS_tot

print(f"\nPREDICTION ERRORS (at experimental time points)")
print(f"  MAE   = {MAE:.3f} mcg/day")
print(f"  RMSE  = {RMSE:.3f} mcg/day")
print(f"  MAPE  = {MAPE:.2f} %")
print(f"  R²    = {R2:.4f}")

# Per-point table
err_df = pd.DataFrame({
    "t_exp (days)":       t_exp,
    "m_exp (mcg/day)":    m_exp,
    "m_sim (mcg/day)":    m_sim_interp,
    "residual (mcg/day)": residuals,
    "abs error (mcg/day)":np.abs(residuals),
    "% error":            np.abs(residuals / m_exp) * 100,
})
print(f"\n{err_df.to_string(index=False, float_format=lambda x: f'{x:.3f}')}")

# =====================================================
# PLOTS: requested multi-panel diagnostic figure
# =====================================================
sim_full = simulate(DOS_fit, PDL_fit, ha_skin_fit, ha_ends_fit, return_details=True)
t = sim_full["t_days"]

# Export the full simulation table so this script also replaces the old
# simulation-only CSV generator.
results_df = pd.DataFrame({
    "t (days)": t,
    "Rd (cm)": sim_full["Rd"],
    "Zd (cm)": sim_full["Zd"],
    "m_skin_burst (mcg/day)": sim_full["m_skin_burst"],
    "m_skin_burst_net (mcg/day)": sim_full["m_skin_burst_net"],
    "m_skin_sustained (mcg/day)": sim_full["m_skin_sustained"],
    "m_skin_combined (mcg/day)": sim_full["m_skin_combined"],
    "m_ends (mcg/day)": sim_full["m_ends"],
    "Q_skin (mg)": sim_full["Q_skin_mg"],
    "Q_ends (mg)": sim_full["Q_ends_mg"],
    "m_skin_effective (mcg/day)": sim_full["m_skin_effective"],
    "Q_skin_effective (mg)": sim_full["Q_skin_effective_mg"],
    "m_ends_effective (mcg/day)": sim_full["m_ends_effective"],
    "Q_ends_effective (mg)": sim_full["Q_ends_effective_mg"],
    "m_total_effective (mcg/day)": sim_full["m_total"],
    "Q_total_effective (mg)": sim_full["Q_total_effective_mg"],
})

results_csv = RESULTS_DIRECTORY / "Paul_model_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"\nFull simulation table saved: {results_csv.name}")

SKIN_COLOR = "#1f77b4"
ENDS_COLOR = "#ff7f0e"
TOTAL_COLOR = "#d62728"
X_MAX_YEARS = int(np.ceil(np.nanmax(t) / 365.0))
X_MAX_DAYS = X_MAX_YEARS * 365
X_TICKS_YEARS_MAJOR = np.arange(0, X_MAX_YEARS + 1, 1.0)
X_TICKS_YEARS_MINOR = np.arange(0, X_MAX_YEARS + 0.5, 0.5)
X_TICKS_DAYS_MAJOR = X_TICKS_YEARS_MAJOR * 365
X_TICKS_DAYS_MINOR = X_TICKS_YEARS_MINOR * 365

def apply_time_axis(ax):
    ax.set_xlim(0, X_MAX_DAYS)
    ax.set_xticks(X_TICKS_DAYS_MAJOR)
    ax.set_xticks(X_TICKS_DAYS_MINOR, minor=True)
    ax.set_xticklabels([f"{int(y)}" for y in X_TICKS_YEARS_MAJOR])
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(which="minor", axis="x", linestyle=":", alpha=0.25)

fig, axes = plt.subplots(3, 2, figsize=(15, 13))
fig.subplots_adjust(hspace=0.42, wspace=0.28)

# 1) Rd, Zd vs time (dual y-axis)
ax = axes[0, 0]
ax_r = ax.twinx()
line_rd, = ax.plot(t, sim_full["Rd"], lw=2, color=SKIN_COLOR, label="Rd")
line_zd, = ax_r.plot(t, sim_full["Zd"], lw=2, color=ENDS_COLOR, ls="-", label="Zd")
ax.set_title("A. Depletion layers thickness vs time")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Rd (cm)", color=SKIN_COLOR)
ax_r.set_ylabel("Zd (cm)", color=ENDS_COLOR)
ax.tick_params(axis="y", colors=SKIN_COLOR)
ax_r.tick_params(axis="y", colors=ENDS_COLOR)
ax.set_ylim(0, 0.1)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
ax.legend([line_rd, line_zd], ["Rd", "Zd"], loc="upper right")
ax.grid(alpha=0.25)
apply_time_axis(ax)

# 2) Cumulative release by pathway
ax = axes[0, 1]
ax.plot(t, sim_full["Q_skin_mg"], lw=2, color=SKIN_COLOR, label="Q_skin")
ax.plot(t, sim_full["Q_ends_mg"], lw=2, color=ENDS_COLOR, label="Q_ends")
ax.set_title("B. Cumulative release by pathway")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Cumulative release (mg)")
ax.legend()
ax.grid(alpha=0.25)
apply_time_axis(ax)

# 3) net burst (first 30 days), m_skin_sustained, m_skin_effective
ax = axes[1, 0]
mask_burst_30d = t <= 30
ax.plot(t[mask_burst_30d], sim_full["m_skin_burst_net"][mask_burst_30d], lw=1.8, color=SKIN_COLOR, ls=":", label="m_skin_burst_net (0–30 d)")
ax.plot(t, sim_full["m_skin_sustained"], lw=1.8, color=SKIN_COLOR, ls="--", label="m_skin_sustained")
ax.plot(t, sim_full["m_skin_effective"], lw=2.2, color=SKIN_COLOR, ls="-", label="m_skin_effective")
ax.set_title("C. Skin release rates")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Release rate (mcg/day)")
ax.legend(loc="upper right")
ax.grid(alpha=0.25)
apply_time_axis(ax)

# 4) m_ends, m_ends_effective (omit t=0 spike)
ax = axes[1, 1]
mask_nonzero_time = t > 0
ax.plot(t[mask_nonzero_time], sim_full["m_ends"][mask_nonzero_time], lw=1.8, color=ENDS_COLOR, ls="--", label="m_ends")
ax.plot(t[mask_nonzero_time], sim_full["m_ends_effective"][mask_nonzero_time], lw=2.2, color=ENDS_COLOR, ls="-", label="m_ends_effective")
ax.set_title("D. Ends release rates (t=0 omitted)")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Release rate (mcg/day)")
ax.legend(loc="upper right")
ax.grid(alpha=0.25)
apply_time_axis(ax)

if np.any(~np.isnan(sim_full["m_ends"])):
    mends_t0 = sim_full["m_ends"][0]
    y_ref_ends = np.nanmax(sim_full["m_ends"][mask_nonzero_time][:5]) if np.any(mask_nonzero_time) else mends_t0
    ax.text(
        6, y_ref_ends * 0.92,
        f"t=0 point omitted: m_ends(0)={mends_t0:.1f} mcg/day\n"
        "due to extremely small diffusion path; no physical meaning.",
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999", alpha=0.9)
    )

# 5) m_total with experimental data (omit t=0 spike)
ax = axes[2, 0]
mask_total_nonzero = t > 0
ax.plot(t[mask_total_nonzero], sim_full["m_total"][mask_total_nonzero], lw=2.4, color=TOTAL_COLOR, label="Simulated m_total (t>0)")
if m_exp_sd is not None and np.any(~np.isnan(m_exp_sd)):
    valid_sd = ~np.isnan(m_exp_sd)
    ax.errorbar(
        t_exp[valid_sd], m_exp[valid_sd], yerr=m_exp_sd[valid_sd],
        fmt="none", ecolor="black", elinewidth=1.0, capsize=3, alpha=0.9, zorder=4
    )
ax.scatter(
    t_exp, m_exp,
    s=40, facecolors="none", edgecolors="black", linewidths=1.0,
    zorder=5, label="Experimental"
)
ax.set_title("E. Total release: simulation vs experiment")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Release rate (mcg/day)")
ax.legend(loc="upper right")
ax.grid(alpha=0.25)
apply_time_axis(ax)
ax.text(
    0.03, 0.97,
    f"MAE = {MAE:.2f} mcg/day\nMAPE = {MAPE:.2f}%\nR² = {R2:.4f}",
    transform=ax.transAxes,
    va="top", ha="left", fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999", alpha=0.95)
)
ax.text(
    0.03, 0.73,
    "MAE: mean absolute error\n"
    "MAPE: mean absolute percentage error",
    transform=ax.transAxes,
    va="top", ha="left", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#bbb", alpha=0.90)
)

# Remaining drug is calculated from the cumulative simulated total release:
# percentage remaining = (Load - cumulative released) / Load × 100.
remaining_annotation_times = [3 * 365, 5 * 365, t[mask_total_nonzero][-1]]
remaining_annotation_labels = ["Year 3", "Year 5", "Model depletion endpoint"]
remaining_text_positions = [
    (3.35 * 365, 61.0),
    (5.20 * 365, 43.0),
    (6.25 * 365, 23.0),
]
q_total_mg = sim_full["Q_total_effective_mg"]
for annotation_day, annotation_label, text_position in zip(
        remaining_annotation_times, remaining_annotation_labels, remaining_text_positions):
    release_rate = np.interp(annotation_day, t[mask_total_nonzero], sim_full["m_total"][mask_total_nonzero])
    released_mg = np.interp(annotation_day, t, q_total_mg)
    remaining_mg = max(Load - released_mg, 0.0)
    remaining_pct = remaining_mg / Load * 100.0
    label_time = annotation_day / 365.0
    if annotation_label == "Model depletion endpoint":
        label_text = (
            f"{annotation_label}\n({label_time:.2f} yr)\n"
            f"Remaining: {remaining_pct:.1f}% ({remaining_mg:.1f} mg)"
        )
    else:
        label_text = f"{annotation_label}\nRemaining: {remaining_pct:.1f}% ({remaining_mg:.1f} mg)"
    ax.annotate(
        label_text,
        xy=(annotation_day, release_rate), xycoords="data",
        xytext=text_position, textcoords="data",
        ha="center", va="bottom", fontsize=7.8,
        arrowprops=dict(arrowstyle="->", color="#555", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#888", alpha=0.92),
        zorder=6,
    )

# 6) Simulation error panel
ax = axes[2, 1]
ax.axhline(0, color="gray", lw=1)
ax.scatter(t_exp, residuals, s=32, color="#8c564b", label="Residual (sim - exp)")
ax.plot(t_exp, residuals, color="#8c564b", alpha=0.6)
ax.set_title("F. Simulation error at experimental time points")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Residual (mcg/day)")
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
apply_time_axis(ax)

fig.suptitle(
    "Figure 1. Paul model fitting diagnostics\n"
    f"DOS={DOS_fit:.3f}, PDL={PDL_fit:.3e} g/(cm·s), ha_skin={ha_skin_fit:.5f} cm, ha_ends={ha_ends_fit:.6f} cm",
    fontsize=14, y=0.995
)
plt.tight_layout(rect=[0, 0.02, 1, 0.97], h_pad=3.0, w_pad=2.5)

# -----------------------------------------------------------------
# PAGE 1b: Zoomed view 0–1 year (diagnose high early error)
# -----------------------------------------------------------------
fig_zoom, axes_zoom = plt.subplots(1, 2, figsize=(15, 6))
fig_zoom.suptitle(
    "Figure 2. Zoomed view: 0–1 year  |  Diagnosing early-time discrepancy\n"
    f"DOS={DOS_fit:.3f}, PDL={PDL_fit:.3e} g/(cm·s), ha_skin={ha_skin_fit:.5f} cm, ha_ends={ha_ends_fit:.6f} cm",
    fontsize=13, y=1.00
)

mask_zoom_t   = (t > 0) & (t / 365 <= 1.0)
mask_zoom_exp = (t_exp > 0) & (t_exp / 365 <= 1.0)
t_zoom_yr     = t[mask_zoom_t] / 365

# --- Left panel: all release components + experiment ---
ax_z = axes_zoom[0]
ax_z.plot(t_zoom_yr, sim_full["m_skin_burst_net"][mask_zoom_t],
          lw=1.6, color=SKIN_COLOR, ls=":", label="m_skin burst net")
ax_z.plot(t_zoom_yr, sim_full["m_skin_sustained"][mask_zoom_t],
          lw=1.6, color=SKIN_COLOR, ls="--", label="m_skin sustained")
ax_z.plot(t_zoom_yr, sim_full["m_skin_effective"][mask_zoom_t],
          lw=2.2, color=SKIN_COLOR, ls="-", label="m_skin effective")
ax_z.plot(t_zoom_yr, sim_full["m_ends_effective"][mask_zoom_t],
          lw=2.2, color=ENDS_COLOR, ls="-", label="m_ends effective")
ax_z.plot(t_zoom_yr, sim_full["m_total"][mask_zoom_t],
          lw=2.8, color=TOTAL_COLOR, ls="-", label="m_total (sim)", zorder=4)

exp_zoom_t  = t_exp[mask_zoom_exp] / 365
exp_zoom_m  = m_exp[mask_zoom_exp]
if m_exp_sd is not None:
    exp_zoom_sd = m_exp_sd[mask_zoom_exp]
    valid_sd_z  = ~np.isnan(exp_zoom_sd)
    if np.any(valid_sd_z):
        ax_z.errorbar(
            exp_zoom_t[valid_sd_z], exp_zoom_m[valid_sd_z], yerr=exp_zoom_sd[valid_sd_z],
            fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=5
        )
ax_z.scatter(exp_zoom_t, exp_zoom_m, s=50, facecolors="none", edgecolors="black",
             linewidths=1.3, zorder=6, label="Experimental (±SD)")
ax_z.set_xlabel("Time (years)", fontsize=11, fontweight="bold")
ax_z.set_ylabel("Release rate (mcg/day)", fontsize=11, fontweight="bold")
ax_z.set_title("A. Release components vs experiment  (0–1 yr)", fontsize=11)
ax_z.set_xlim([0, 1.0])
ax_z.legend(fontsize=9, loc="upper right")
ax_z.grid(alpha=0.25)

# --- Right panel: residual zoomed ---
ax_r = axes_zoom[1]
ax_r.axhline(0, color="gray", lw=1.2, ls="--")
ax_r.scatter(t_exp[mask_zoom_exp] / 365, residuals[mask_zoom_exp],
             s=50, color="#8c564b", zorder=4, label="Residual (sim − exp)")
ax_r.plot(t_exp[mask_zoom_exp] / 365, residuals[mask_zoom_exp],
          color="#8c564b", alpha=0.6, lw=1.5)
ax_r.fill_between(t_exp[mask_zoom_exp] / 365, residuals[mask_zoom_exp], 0,
                  where=residuals[mask_zoom_exp] > 0, alpha=0.15, color="red",
                  label="Overprediction")
ax_r.fill_between(t_exp[mask_zoom_exp] / 365, residuals[mask_zoom_exp], 0,
                  where=residuals[mask_zoom_exp] < 0, alpha=0.15, color="blue",
                  label="Underprediction")
ax_r.set_xlabel("Time (years)", fontsize=11, fontweight="bold")
ax_r.set_ylabel("Residual (sim − exp)  mcg/day", fontsize=11, fontweight="bold")
ax_r.set_title("B. Prediction error  (0–1 yr)", fontsize=11)
ax_r.set_xlim([0, 1.0])
ax_r.legend(fontsize=9, loc="upper right")
ax_r.grid(alpha=0.25)

plt.tight_layout()

# -----------------------------------------------------------------
# PAGE 2: Parameter tables
# -----------------------------------------------------------------
fig2 = plt.figure(figsize=(15, 11))
fig2.suptitle("Parameter summary tables", fontsize=15, y=0.98)

# Top table: short-term data fitted parameters
ax_fit = fig2.add_axes([0.05, 0.68, 0.90, 0.24])
ax_fit.axis("off")
ax_fit.text(0.0, 1.05, "Table 1. Short-term data fitted parameters", fontsize=12, fontweight="bold", transform=ax_fit.transAxes)

fitted_rows = [
    ["DOS", f"{DOS_fit:.6f}", "-", "Degree of supersaturation"],
    ["PDL", f"{PDL_fit:.6e}", "g/(cm·s)", "Fitted permeability of depletion layer"],
    ["ha_skin", f"{ha_skin_fit:.6f}", "cm", "Skin water-layer thickness"],
    ["ha_ends", f"{ha_ends_fit:.6f}", "cm", "Ends water-layer thickness"],
]

tbl_fit = ax_fit.table(
    cellText=fitted_rows,
    colLabels=["Parameter", "Value", "Unit", "Description"],
    cellLoc="left",
    loc="upper left",
    bbox=[0, 0.02, 1.0, 0.88],
)
tbl_fit.auto_set_font_size(False)
tbl_fit.set_fontsize(10)
tbl_fit.scale(1, 1.3)

# Bottom table: directly measured / fixed parameters grouped by category
ax_fix = fig2.add_axes([0.05, 0.06, 0.90, 0.57])
ax_fix.axis("off")
ax_fix.text(0.0, 1.03, "Table 2. Directly measured / fixed parameters (categorized)", fontsize=12, fontweight="bold", transform=ax_fix.transAxes)

fixed_rows = [
    ["Implant geometry", "L", f"{L:.6g}", "cm", "Implant length"],
    ["Implant geometry", "R_i", f"{R_i:.6g}", "cm", "Inner radius"],
    ["Implant geometry", "R_o", f"{R_o:.6g}", "cm", "Outer radius"],
    ["Implant geometry", "V_core", f"{V_core:.6g}", "cm³", "Core volume"],
    ["Implant geometry", "SA_ends", f"{SA_ends:.6g}", "cm²", "Ends surface area"],

    ["Etonogestrel transport", "Cs_28", f"{Cs_28:.6g}", "g/cm³", "Solubility in EVA 28"],
    ["Etonogestrel transport", "D_28", f"{D_28:.6g}", "cm²/s", "Diffusivity in EVA 28"],
    ["Etonogestrel transport", "Cs_DL", f"{Cs_DL:.6g}", "g/cm³", "Solubility in depletion layer"],
    ["Etonogestrel transport", "D_DL", f"{D_DL:.6g}", "cm²/s", "Diffusivity in depletion layer"],
    ["Etonogestrel transport", "Cs_15", f"{Cs_15:.6g}", "g/cm³", "Solubility in EVA 15"],
    ["Etonogestrel transport", "D_15", f"{D_15:.6g}", "cm²/s", "Diffusivity in EVA 15"],
    ["Etonogestrel transport", "Cs_a", f"{Cs_a:.6g}", "g/cm³", "Solubility in water layer"],
    ["Etonogestrel transport", "D_a", f"{D_a:.6g}", "cm²/s", "Diffusivity in water layer"],
    ["Etonogestrel transport", "P_DL reference", f"{P_DL:.6g}", "g/(cm·s)", "Measured reference used to scale fitted PDL"],
    ["Etonogestrel transport", "P_15", f"{P_15:.6g}", "g/(cm·s)", "Permeability of EVA 15"],
    ["Etonogestrel transport", "P_a", f"{P_a:.6g}", "g/(cm·s)", "Permeability of water layer"],

    ["Initial conditions", "Load", f"{Load:.6g}", "mg", "Initial drug load"],
    ["Initial conditions", "C0", f"{C0:.6g}", "g/cm³", "Initial concentration in core"],

    ["Dissolution medium", "sink_condition", f"{sink_condition:.6g}", "-", "C_media / Cs_a"],
    ["Dissolution medium", "C_media", f"{C_media:.6g}", "g/cm³", "Medium concentration"],

    ["Release baseline", "zero_order_ends", f"{zero_order_ends:.6g}", "mcg/day", "Constant baseline release from ends"],
]

tbl_fix = ax_fix.table(
    cellText=fixed_rows,
    colLabels=["Category", "Parameter", "Value", "Unit", "Description"],
    cellLoc="left",
    loc="upper left",
    bbox=[0, 0.0, 1.0, 0.98],
)
tbl_fix.auto_set_font_size(False)
tbl_fix.set_fontsize(9)
tbl_fix.scale(1, 1.2)

# -----------------------------------------------------------------
# FIGURE 8: PDL sensitivity around the fitted full-data value
# -----------------------------------------------------------------
TESTING_WINDOW_START_DAY = 90.0
TESTING_WINDOW_END_DAY = 3.5 * 365.0


def annotate_largest_rate_difference(ax, profiles):
    """Mark the largest departure during the practical 3-month to 3.5-year window.

    ``profiles`` are ordered as -15%, reference, +15%.  Comparisons are
    limited to the 3-month to 3.5-year testing window and to the time range
    shared by each varied profile and the reference.  This avoids comparing a
    depleted profile with extrapolated data.
    """
    reference_t, reference_rate = profiles[1]
    candidates = []

    for profile_index in (0, 2):
        t_case, rate_case = profiles[profile_index]
        shared_end = min(np.nanmax(t_case), np.nanmax(reference_t),
                         TESTING_WINDOW_END_DAY)
        ref_at_case = np.interp(t_case, reference_t, reference_rate)
        valid = (
            (t_case >= TESTING_WINDOW_START_DAY) & (t_case <= shared_end) &
            np.isfinite(rate_case) & np.isfinite(ref_at_case) &
            (ref_at_case > 0)
        )
        if not np.any(valid):
            continue
        candidate_indices = np.where(valid)[0]
        higher_indices = candidate_indices[
            (rate_case[candidate_indices] - ref_at_case[candidate_indices]) > 0
        ]
        if higher_indices.size == 0:
            continue
        rate_increase = rate_case[higher_indices] - ref_at_case[higher_indices]
        idx = higher_indices[np.argmax(rate_increase)]
        candidates.append((rate_case[idx] - ref_at_case[idx], t_case[idx],
                           rate_case[idx], ref_at_case[idx]))

    if not candidates:
        return

    _, t_day, rate, reference = max(candidates, key=lambda item: item[0])
    percent_difference = 100.0 * (rate - reference) / reference
    ax.annotate(
        f"Maximum difference from reference\n(3 months to 3.5 years):\n"
        f"Day {t_day:.0f} ({t_day / 365.0:.2f} years)\n"
        f"{percent_difference:.1f}% higher",
        xy=(t_day / 365.0, rate), xycoords="data",
        xytext=(0.50, 0.50), textcoords="axes fraction",
        ha="left", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#555", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.1),
        zorder=8,
    )


fig8, ax8 = plt.subplots(1, 1, figsize=(10, 6))
pdl_sensitivity_cases = [
    ("PDL -15%", 0.85, "#1f77b4"),
    ("Fitted PDL", 1.00, "#d62728"),
    ("PDL +15%", 1.15, "#ff7f0e"),
]
pdl_sensitivity_profiles = []

for label, factor, color in pdl_sensitivity_cases:
    pdl_case = PDL_fit * factor
    t_case, m_case = simulate(DOS_fit, pdl_case, ha_skin_fit, ha_ends_fit)
    valid_case = (t_case > 0) & ~np.isnan(m_case)
    pdl_sensitivity_profiles.append((t_case, m_case))
    ax8.plot(
        t_case[valid_case] / 365,
        m_case[valid_case],
        linewidth=2.0,
        color=color,
        label=f"{label}: {pdl_case:.2e} g/(cm·s)",
        zorder=4,
    )

if m_exp_sd is not None and np.any(~np.isnan(m_exp_sd)):
    valid_sd = ~np.isnan(m_exp_sd)
    ax8.errorbar(
        t_exp[valid_sd] / 365, m_exp[valid_sd], yerr=m_exp_sd[valid_sd],
        fmt="o", color="black", markersize=5, capsize=3, linewidth=1.0,
        label="Experimental data (±SD)", zorder=3, ecolor="black", alpha=0.85,
    )
    ax8.scatter(
        t_exp[~valid_sd] / 365, m_exp[~valid_sd],
        s=32, alpha=0.75, color="black", zorder=3, edgecolors="black", linewidth=0.5,
    )
else:
    ax8.scatter(
        t_exp / 365, m_exp, s=40, alpha=0.75, color="black",
        label="Experimental data", zorder=3, edgecolors="black", linewidth=0.5,
    )

pdl_sensitivity_max_years = int(np.ceil(max(
    np.nanmax(t_case) for t_case, _ in pdl_sensitivity_profiles
) / 365.0))
ax8.set_xlim([0, pdl_sensitivity_max_years])
ax8.set_xlabel("Time (years)", fontsize=11, fontweight="bold")
ax8.set_ylabel("Release Rate (mcg/day)", fontsize=11, fontweight="bold")
ax8.set_title("Figure 8. Sensitivity of drug release profiles to fitted PDL ±15%", fontsize=12, fontweight="bold")
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=9, loc="upper right")
plt.tight_layout()

# -----------------------------------------------------------------
# FIGURES 9–10: Pskin and implant-skin-thickness sensitivities
# -----------------------------------------------------------------
def create_sensitivity_figure(figure_number, title, cases, note,
                              annotate_max_difference=False, show_note=True):
    """Plot full-data profiles for a set of one-factor sensitivity cases."""
    fig_sensitivity, ax_sensitivity = plt.subplots(1, 1, figsize=(10, 6))
    profiles = []

    for label, color, simulation_kwargs in cases:
        case_kwargs = dict(simulation_kwargs)
        ha_skin_case = case_kwargs.pop("ha_skin_override", ha_skin_fit)
        pdl_case = case_kwargs.pop("PDL_override", PDL_fit)
        t_case, m_case = simulate(
            DOS_fit, pdl_case, ha_skin_case, ha_ends_fit, **case_kwargs
        )
        valid_case = (t_case > 0) & ~np.isnan(m_case)
        profiles.append((t_case, m_case))
        ax_sensitivity.plot(
            t_case[valid_case] / 365, m_case[valid_case], linewidth=2.0,
            color=color, label=label, zorder=4,
        )

    if m_exp_sd is not None and np.any(~np.isnan(m_exp_sd)):
        valid_sd = ~np.isnan(m_exp_sd)
        ax_sensitivity.errorbar(
            t_exp[valid_sd] / 365, m_exp[valid_sd], yerr=m_exp_sd[valid_sd],
            fmt="o", color="black", markersize=5, capsize=3, linewidth=1.0,
            label="Experimental data (±SD)", zorder=3, ecolor="black", alpha=0.85,
        )
        ax_sensitivity.scatter(
            t_exp[~valid_sd] / 365, m_exp[~valid_sd], s=32, alpha=0.75,
            color="black", zorder=3, edgecolors="black", linewidth=0.5,
        )
    else:
        ax_sensitivity.scatter(
            t_exp / 365, m_exp, s=40, alpha=0.75, color="black",
            label="Experimental data", zorder=3, edgecolors="black", linewidth=0.5,
        )

    x_max_years = int(np.ceil(max(
        np.nanmax(t_case) for t_case, _ in profiles
    ) / 365.0))
    ax_sensitivity.set_xlim([0, x_max_years])
    ax_sensitivity.set_xlabel("Time (years)", fontsize=11, fontweight="bold")
    ax_sensitivity.set_ylabel("Release Rate (mcg/day)", fontsize=11, fontweight="bold")
    ax_sensitivity.set_title(
        f"Figure {figure_number}. {title}", fontsize=12, fontweight="bold"
    )
    ax_sensitivity.grid(True, alpha=0.3)
    ax_sensitivity.legend(fontsize=9, loc="upper right")
    if show_note:
        ax_sensitivity.text(
            0.02, 0.67, note, transform=ax_sensitivity.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999", alpha=0.9),
        )
    if annotate_max_difference:
        annotate_largest_rate_difference(ax_sensitivity, profiles)
    fig_sensitivity.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)
    return fig_sensitivity


def create_first_three_months_figure(figure_number, title, cases, note):
    """Create a 0-3-month view of a three-profile sensitivity analysis."""
    fig_early, ax_early = plt.subplots(1, 1, figsize=(10, 6))

    for label, color, simulation_kwargs in cases:
        case_kwargs = dict(simulation_kwargs)
        ha_skin_case = case_kwargs.pop("ha_skin_override", ha_skin_fit)
        pdl_case = case_kwargs.pop("PDL_override", PDL_fit)
        t_case, m_case = simulate(
            DOS_fit, pdl_case, ha_skin_case, ha_ends_fit, **case_kwargs
        )
        valid_case = (t_case > 0) & (t_case <= TESTING_WINDOW_START_DAY) & ~np.isnan(m_case)
        ax_early.plot(
            t_case[valid_case] / 30.4375, m_case[valid_case], linewidth=2.0,
            color=color, label=label, zorder=4,
        )

    early_data = t_exp <= TESTING_WINDOW_START_DAY
    if m_exp_sd is not None and np.any(~np.isnan(m_exp_sd[early_data])):
        valid_sd = early_data & ~np.isnan(m_exp_sd)
        ax_early.errorbar(
            t_exp[valid_sd] / 30.4375, m_exp[valid_sd], yerr=m_exp_sd[valid_sd],
            fmt="o", color="black", markersize=5, capsize=3, linewidth=1.0,
            label="Experimental data (±SD)", zorder=3, ecolor="black", alpha=0.85,
        )
        no_sd = early_data & np.isnan(m_exp_sd)
        ax_early.scatter(
            t_exp[no_sd] / 30.4375, m_exp[no_sd], s=32, alpha=0.75,
            color="black", zorder=3, edgecolors="black", linewidth=0.5,
        )
    else:
        ax_early.scatter(
            t_exp[early_data] / 30.4375, m_exp[early_data], s=40, alpha=0.75,
            color="black", label="Experimental data", zorder=3,
            edgecolors="black", linewidth=0.5,
        )

    ax_early.set_xlim([0, 3])
    ax_early.set_xticks(np.arange(0, 4, 1))
    ax_early.set_xlabel("Time (months)", fontsize=11, fontweight="bold")
    ax_early.set_ylabel("Release Rate (mcg/day)", fontsize=11, fontweight="bold")
    ax_early.set_title(
        f"Figure {figure_number}. {title} - first 3 months",
        fontsize=12, fontweight="bold",
    )
    ax_early.grid(True, alpha=0.3)
    ax_early.legend(fontsize=9, loc="upper right")
    fig_early.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)
    return fig_early


fig8_1 = create_first_three_months_figure(
    "8-1",
    "Sensitivity of drug release profiles to fitted PDL ±15%",
    [
        (f"PDL -15%: {PDL_fit * 0.85:.2e} g/(cm·s)", "#1f77b4", {"PDL_override": PDL_fit * 0.85}),
        (f"Fitted PDL: {PDL_fit:.2e} g/(cm·s)", "#d62728", {"PDL_override": PDL_fit}),
        (f"PDL +15%: {PDL_fit * 1.15:.2e} g/(cm·s)", "#ff7f0e", {"PDL_override": PDL_fit * 1.15}),
    ],
    "DOS, ha_skin, and ha_ends held at\ntheir full-data fitted values.",
)


fig9 = create_sensitivity_figure(
    9,
    "Sensitivity of drug release profiles to reference Pskin (P_15) ±15%",
    [
        (f"Pskin -15%: {P_15 * 0.85:.2e} g/(cm·s)", "#1f77b4", {"P15_factor": 0.85}),
        (f"Reference Pskin: {P_15:.2e} g/(cm·s)", "#d62728", {"P15_factor": 1.00}),
        (f"Pskin +15%: {P_15 * 1.15:.2e} g/(cm·s)", "#ff7f0e", {"P15_factor": 1.15}),
    ],
    "PDL, DOS, ha_skin, and ha_ends held at\ntheir full-data fitted values.",
    annotate_max_difference=False,
    show_note=False,
)

fig9_1 = create_first_three_months_figure(
    "9-1",
    "Sensitivity of drug release profiles to reference Pskin (P_15) ±15%",
    [
        (f"Pskin -15%: {P_15 * 0.85:.2e} g/(cm·s)", "#1f77b4", {"P15_factor": 0.85}),
        (f"Reference Pskin: {P_15:.2e} g/(cm·s)", "#d62728", {"P15_factor": 1.00}),
        (f"Pskin +15%: {P_15 * 1.15:.2e} g/(cm·s)", "#ff7f0e", {"P15_factor": 1.15}),
    ],
    "PDL, DOS, ha_skin, and ha_ends held at\ntheir full-data fitted values.",
)

fig10 = create_sensitivity_figure(
    10,
    "Sensitivity of drug release profiles to lskin = R_o - R_i",
    [
        ("lskin = 50 µm (R_i = 0.095 cm)", "#1f77b4", {"Ri_override": 0.095}),
        ("lskin = 60 µm (R_i = 0.094 cm)", "#d62728", {"Ri_override": 0.094}),
        ("lskin = 70 µm (R_i = 0.093 cm)", "#ff7f0e", {"Ri_override": 0.093}),
    ],
    "R_o fixed at 0.100 cm; PDL, DOS, ha_skin, and\nha_ends held at their full-data fitted values.",
)

fig11 = create_sensitivity_figure(
    11,
    "Sensitivity of drug release profiles to fitted ha_skin ±25%",
    [
        (f"ha_skin -25%: {ha_skin_fit * 0.75:.4f} cm", "#1f77b4", {"ha_skin_override": ha_skin_fit * 0.75}),
        (f"Fitted ha_skin: {ha_skin_fit:.4f} cm", "#d62728", {"ha_skin_override": ha_skin_fit}),
        (f"ha_skin +25%: {ha_skin_fit * 1.25:.4f} cm", "#ff7f0e", {"ha_skin_override": ha_skin_fit * 1.25}),
    ],
    "PDL, DOS, Pskin (P_15), ha_ends, and\nimplant geometry held at their fitted values.",
)

ha_skin_low = ha_skin_fit * 0.75
ha_skin_high = ha_skin_fit * 1.25

fig12 = create_sensitivity_figure(
    12,
    f"Pskin (P_15) sensitivity at ha_skin -25% ({ha_skin_low:.4f} cm)",
    [
        (f"Pskin -15%: {P_15 * 0.85:.2e} g/(cm·s)", "#1f77b4", {"ha_skin_override": ha_skin_low, "P15_factor": 0.85}),
        (f"Reference Pskin: {P_15:.2e} g/(cm·s)", "#d62728", {"ha_skin_override": ha_skin_low, "P15_factor": 1.00}),
        (f"Pskin +15%: {P_15 * 1.15:.2e} g/(cm·s)", "#ff7f0e", {"ha_skin_override": ha_skin_low, "P15_factor": 1.15}),
    ],
    "ha_skin fixed at -25%; PDL, DOS, ha_ends,\nand implant geometry held at their fitted values.",
    annotate_max_difference=True,
)

fig13 = create_sensitivity_figure(
    13,
    f"PDL sensitivity at ha_skin -25% ({ha_skin_low:.4f} cm)",
    [
        (f"PDL -15%: {PDL_fit * 0.85:.2e} g/(cm·s)", "#1f77b4", {"ha_skin_override": ha_skin_low, "PDL_override": PDL_fit * 0.85}),
        (f"Fitted PDL: {PDL_fit:.2e} g/(cm·s)", "#d62728", {"ha_skin_override": ha_skin_low, "PDL_override": PDL_fit}),
        (f"PDL +15%: {PDL_fit * 1.15:.2e} g/(cm·s)", "#ff7f0e", {"ha_skin_override": ha_skin_low, "PDL_override": PDL_fit * 1.15}),
    ],
    "ha_skin fixed at -25%; DOS, Pskin (P_15), ha_ends,\nand implant geometry held at their fitted values.",
    annotate_max_difference=True,
)

fig14 = create_sensitivity_figure(
    14,
    f"Pskin (P_15) sensitivity at ha_skin +25% ({ha_skin_high:.4f} cm)",
    [
        (f"Pskin -15%: {P_15 * 0.85:.2e} g/(cm·s)", "#1f77b4", {"ha_skin_override": ha_skin_high, "P15_factor": 0.85}),
        (f"Reference Pskin: {P_15:.2e} g/(cm·s)", "#d62728", {"ha_skin_override": ha_skin_high, "P15_factor": 1.00}),
        (f"Pskin +15%: {P_15 * 1.15:.2e} g/(cm·s)", "#ff7f0e", {"ha_skin_override": ha_skin_high, "P15_factor": 1.15}),
    ],
    "ha_skin fixed at +25%; PDL, DOS, ha_ends,\nand implant geometry held at their fitted values.",
)

fig15 = create_sensitivity_figure(
    15,
    f"PDL sensitivity at ha_skin +25% ({ha_skin_high:.4f} cm)",
    [
        (f"PDL -15%: {PDL_fit * 0.85:.2e} g/(cm·s)", "#1f77b4", {"ha_skin_override": ha_skin_high, "PDL_override": PDL_fit * 0.85}),
        (f"Fitted PDL: {PDL_fit:.2e} g/(cm·s)", "#d62728", {"ha_skin_override": ha_skin_high, "PDL_override": PDL_fit}),
        (f"PDL +15%: {PDL_fit * 1.15:.2e} g/(cm·s)", "#ff7f0e", {"ha_skin_override": ha_skin_high, "PDL_override": PDL_fit * 1.15}),
    ],
    "ha_skin fixed at +25%; DOS, Pskin (P_15), ha_ends,\nand implant geometry held at their fitted values.",
)

# -----------------------------------------------------------------
# FIGURE 16: Crossed PDL/Pskin-ratio sensitivity
# -----------------------------------------------------------------
fig16, ax16 = plt.subplots(1, 1, figsize=(10, 6))
crossed_ratio_cases = [
    ("Reference (PDL 100%, Pskin 100%)", PDL_fit, 1.00, "#d62728"),
    ("PDL 85%, Pskin 115%", PDL_fit * 0.85, 1.15, "#1f77b4"),
    ("PDL 115%, Pskin 85%", PDL_fit * 1.15, 0.85, "#ff7f0e"),
]
crossed_ratio_profiles = []

for case_index, (case_name, pdl_case, pskin_factor, color) in enumerate(crossed_ratio_cases):
    t_case, m_case = simulate(
        DOS_fit, pdl_case, ha_skin_fit, ha_ends_fit, P15_factor=pskin_factor
    )
    valid_case = (t_case > 0) & ~np.isnan(m_case)
    depletion_years = t_case[-1] / 365.0
    pskin_case = P_15 * pskin_factor
    ratio_case = pskin_case / pdl_case
    crossed_ratio_profiles.append((t_case, m_case, depletion_years))
    ax16.plot(
        t_case[valid_case] / 365.0, m_case[valid_case], linewidth=2.2,
        color=color,
        label=(f"{case_name}\n"
               f"PDL={pdl_case:.2e}; Pskin={pskin_case:.2e}; Pskin/PDL={ratio_case:.3f}\n"
               f"Total depletion={depletion_years:.2f} years"),
        zorder=4,
    )
    ax16.annotate(
        f"t_dep={depletion_years:.2f} y",
        xy=(depletion_years, m_case[-1]), xycoords="data",
        xytext=(5, 14 + 16 * case_index), textcoords="offset points",
        color=color, fontsize=8.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        zorder=6,
    )

max_depletion_years = max(profile[2] for profile in crossed_ratio_profiles)
ax16.set_xlim([0, int(np.ceil(max_depletion_years)) + 1])
ax16.set_xlabel("Time (years)", fontsize=11, fontweight="bold")
ax16.set_ylabel("Release Rate (mcg/day)", fontsize=11, fontweight="bold")
ax16.set_title(
    "Figure 16. Impact of crossed PDL/Pskin changes on drug release profiles",
    fontsize=12, fontweight="bold",
)
ax16.grid(True, alpha=0.3)
ax16.legend(fontsize=8, loc="upper right", framealpha=0.95)
fig16.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)

fig16_1, ax16_1 = plt.subplots(1, 1, figsize=(10, 6))
for (case_name, pdl_case, pskin_factor, color), (t_case, m_case, _) in zip(
        crossed_ratio_cases, crossed_ratio_profiles):
    valid_case = (t_case > 0) & (t_case <= 90.0) & ~np.isnan(m_case)
    ax16_1.plot(
        t_case[valid_case] / 30.4375, m_case[valid_case], linewidth=2.2,
        color=color,
        label=(f"{case_name}\n"
               f"PDL={pdl_case:.2e}; Pskin={P_15 * pskin_factor:.2e}"),
        zorder=4,
    )

ax16_1.set_xlim([0, 3])
ax16_1.set_xticks(np.arange(0, 4, 1))
ax16_1.set_xlabel("Time (months)", fontsize=11, fontweight="bold")
ax16_1.set_ylabel("Release Rate (mcg/day)", fontsize=11, fontweight="bold")
ax16_1.set_title(
    "Figure 16-1. Crossed PDL/Pskin changes - first 3 months",
    fontsize=12, fontweight="bold",
)
ax16_1.grid(True, alpha=0.3)
ax16_1.legend(fontsize=9, loc="upper right", framealpha=0.95)
fig16_1.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)


def create_model_audit_pages(simulation):
    """Create equation/variable and quarterly-output pages for the PDF appendix."""
    pages = []

    # Equation pages deliberately use the same fitted model quantities exported
    # in Paul_model_results.csv, so the report is auditable from equations to data.
    fig_eq1 = plt.figure(figsize=(15, 11), facecolor="white")
    ax_eq1 = fig_eq1.add_axes([0, 0, 1, 1])
    ax_eq1.axis("off")
    fig_eq1.text(0.05, 0.90, "Appendix A. Release-model equations (1 of 2)",
                 fontsize=18, fontweight="bold", va="top")
    fig_eq1.text(0.05, 0.855,
                 "Rates are calculated in g/s and then converted to mcg/day (×10⁶ × 86,400). "
                 "The equations below use the fitted values of DOS, PDL, ha_skin, and ha_ends.",
                 fontsize=10, va="top")
    equation_lines_1 = [
        ("1. Burst release through the skin (first 30 days)", 11.5, "normal"),
        (r"$R_{a,skin}=R_o+h_{a,skin}, \quad C_{core,0}=DOS\,C_{s,28}, \quad P_{15}=C_{s,15}D_{15}, \quad P_a=C_{s,a}D_a$", 10.5, "normal"),
        (r"$C_{1,burst}=\dfrac{2\pi L}{\ln(R_o/R_i)/P_{15}+\ln(R_{a,skin}/R_o)/P_a}$", 10.5, "normal"),
        (r"$C_{2,burst}=1-C_{media}/C_{s,a}, \quad C_{3,burst}=C_{core,0}/C_{s,28}-1$", 10.5, "normal"),
        (r"$m_{burst}(t)=C_{1,burst}[C_{2,burst}+C_{3,burst}\exp\{-C_{1,burst}t/(C_{s,28}V_{core})\}]$", 10.5, "normal"),
        ("", 8, "normal"),
        ("2. Radial depletion position used for sustained skin release", 11.5, "normal"),
        (r"$C_{1,R_d}=-(1-s)/(C_0-C_{s,DL}), \quad C_{2,R_d}=1/PDL$", 10.5, "normal"),
        (r"$C_{4,R_d}=\ln(R_o/R_i)/P_{15}+\ln(R_{a,skin}/R_o)/P_a$", 10.5, "normal"),
        (r"$A=C_{2,R_d}(2\ln R_i+1)+2C_{4,R_d}$", 10.5, "normal"),
        (r"$t(R_d)=\dfrac{(R_d^2-R_i^2)A+2C_{2,R_d}[-R_d^2\ln R_d+R_i^2\ln R_i]}{4C_{1,R_d}}$", 10.5, "normal"),
        ("", 8, "normal"),
        ("3. Sustained skin release", 11.5, "normal"),
        (r"$m_{skin,sus}(t)=\dfrac{2\pi L(1-s)}{\ln(R_i/R_d(t))/PDL+\ln(R_o/R_i)/P_{15}+\ln(R_{a,skin}/R_o)/P_a}$", 10.5, "normal"),
    ]
    line_y = 0.775
    for line, fontsize, weight in equation_lines_1:
        if line:
            fig_eq1.text(0.06, line_y, line, fontsize=fontsize, fontweight=weight, va="top")
        line_y -= 0.044 if line else 0.022
    fig_eq1.text(0.06, 0.145,
                 "At each simulated time, R_d is obtained by numerically solving the displayed t(R_d) relation. "
                 "The model includes radius-spaced points from 5% R_i to 0.001% R_i to resolve late depletion.",
                 fontsize=10, va="top", wrap=True)
    pages.append(fig_eq1)

    fig_eq2 = plt.figure(figsize=(15, 11), facecolor="white")
    ax_eq2 = fig_eq2.add_axes([0, 0, 1, 1])
    ax_eq2.axis("off")
    fig_eq2.text(0.05, 0.95, "Appendix A. Release-model equations and variables (2 of 2)",
                 fontsize=18, fontweight="bold", va="top")
    equation_lines_2 = [
        ("4. End release", 11.5),
        (r"$C_{1,end}=-(1-s)/(C_0-C_{s,DL}), \quad C_{2,end}=L/(2PDL)+h_{a,end}/P_a, \quad C_{3,end}=-1/PDL$", 10.5),
        (r"$t=\dfrac{0.5C_{3,end}Z_d^2+C_{2,end}Z_d-C_{2,end}L/2-C_{3,end}L^2/8}{C_{1,end}}$", 10.5),
        (r"$m_{end}(t)=\dfrac{(1-s)SA_{ends}}{[L/2-Z_d(t)]/PDL+h_{a,end}/P_a}+m_{end,0}$", 10.5),
        ("", 8),
        ("5. Combining burst, skin, and ends release", 11.5),
        (r"$m_{burst,net}=\max\{m_{burst}-m_{skin,sus}-(t/30)m_{burst,extra}(30),0\}$ for $t\leq30$ days; otherwise 0", 10.0),
        (r"$m_{skin}=m_{skin,sus}+m_{burst,net}, \quad Q_j(t)=\int_0^t m_j(\tau)\,d\tau$ (numerical trapezoidal integration)", 10.0),
        (r"$m_{skin,eff}=m_{skin}\max(1-Q_{end}/Load,0), \quad m_{end,eff}=m_{end}\max(1-Q_{skin}/Load,0)$", 10.0),
        (r"$m_{total}=m_{skin,eff}+m_{end,eff}, \quad Q_{total}=\int_0^t m_{total}(\tau)\,d\tau$", 10.0),
    ]
    line_y = 0.88
    for line, fontsize in equation_lines_2:
        if line:
            fig_eq2.text(0.06, line_y, line, fontsize=fontsize, va="top")
        line_y -= 0.038 if line else 0.018
    fig_eq2.text(0.06, 0.55, "Variables and fixed inputs", fontsize=13, fontweight="bold", va="top")
    variable_columns = [
        "Geometry: L = implant length; R_i/R_o = inner/outer radius; R_a,skin = outer aqueous boundary; R_d = radial depletion front; Z_d = end depletion front; V_core = πR_i²L; SA_ends = 2πR_i²; h_a,skin and h_a,end = aqueous boundary-layer thicknesses.",
        "Transport/concentration: C_s,28, C_s,DL, C_s,15, C_s,a = solubilities; D_28, D_DL, D_15, D_a = diffusivities; PDL = fitted drug-layer permeability; P_15 = Pskin = C_s,15D_15; P_a = C_s,aD_a; C_0 = initial core concentration; C_media = sC_s,a; s = sink condition.",
        "Release/output: DOS = fitted initial concentration multiplier; t = time (s in equations, days in output); m = instantaneous release rate; Q = cumulative released amount; Load = 68 mg initial drug load; m_end,0 = zero-order end term (0 mcg/day here); subscripts burst, skin, end, sus, eff, and total indicate the corresponding component."
    ]
    x_positions = [0.06, 0.365, 0.67]
    for x_pos, text_block in zip(x_positions, variable_columns):
        fig_eq2.text(x_pos, 0.50, textwrap.fill(text_block, width=48), fontsize=9.5, va="top",
                     bbox=dict(boxstyle="round,pad=0.55", facecolor="#f6f6f6", edgecolor="#bdbdbd"))
    fig_eq2.text(0.06, 0.17,
                 "The effective rates prevent either pathway from releasing more drug than remains after the other pathway's cumulative release. "
                 "Thus the table reports Q_skin,eff + Q_end,eff = Q_total at every time point (within numerical precision).",
                 fontsize=10, va="top", wrap=True)
    pages.append(fig_eq2)

    # Quarterly reporting matches the 30-day late-stage simulation grid while
    # keeping the audit table compact. Add the exact final depletion time.
    t_days = simulation["t_days"]
    final_month = t_days[-1] / 30.4375
    report_months = np.arange(3.0, final_month - 1e-8, 3.0)
    if len(report_months) == 0 or not np.isclose(report_months[-1], final_month):
        report_months = np.append(report_months, final_month)
    report_days = report_months * 30.4375

    values = [
        ("Skin release rate (mcg/day)", simulation["m_skin_combined"]),
        ("Effective skin release rate (mcg/day)", simulation["m_skin_effective"]),
        ("Skin release drug amount (mg)", simulation["Q_skin_effective_mg"]),
        ("Ends release rate (mcg/day)", simulation["m_ends"]),
        ("Effective ends release rate (mcg/day)", simulation["m_ends_effective"]),
        ("Ends release drug amount (mg)", simulation["Q_ends_effective_mg"]),
        ("Total release rate (mcg/day)", simulation["m_total"]),
        ("Total release drug amount (mg)", simulation["Q_total_effective_mg"]),
    ]
    headers = ["Time\n(months)"] + [name.replace(" (mcg/day)", "\n(mcg/day)").replace(" (mg)", "\n(mg)") for name, _ in values]
    table_rows = []
    for month, day in zip(report_months, report_days):
        row = [f"{month:.2f}" if np.isclose(month, final_month) else f"{month:.0f}"]
        for name, series in values:
            value = np.interp(day, t_days, series)
            row.append(f"{value:.3f}")
        table_rows.append(row)

    for page_number, start in enumerate(range(0, len(table_rows), 16), start=1):
        subset = table_rows[start:start + 16]
        fig_table, ax_table = plt.subplots(figsize=(15, 11), facecolor="white")
        ax_table.axis("off")
        ax_table.set_title(
            f"Appendix B. Quarterly simulated release data ({page_number} of {int(np.ceil(len(table_rows) / 16))})",
            fontsize=17, fontweight="bold", pad=24,
        )
        ax_table.text(0.0, 1.01,
                      "Rates are instantaneous fitted-model rates. Drug amounts are cumulative effective released mass; the final row is the simulated depletion endpoint.",
                      transform=ax_table.transAxes, fontsize=9.5, va="bottom")
        table = ax_table.table(cellText=subset, colLabels=headers, cellLoc="center", colLoc="center",
                               loc="upper center", bbox=[0.01, 0.10, 0.98, 0.83])
        table.auto_set_font_size(False)
        table.set_fontsize(6.7)
        table.scale(1.0, 1.45)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#cfcfcf")
            if row == 0:
                cell.set_facecolor("#1f4e79")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f3f7fa")
        ax_table.text(0.01, 0.035,
                      "Skin release rate includes sustained release plus the tapered 0–30-day burst contribution. "
                      "Effective rates apply the cross-pathway depletion factor used by the model.",
                      transform=ax_table.transAxes, fontsize=9, va="bottom")
        pages.append(fig_table)

    return pages


def create_medium_concentration_appendix_page(simulation):
    """Calculate 0-5 year fitted AUC and daily-replacement medium concentration."""
    analysis_days = 5 * 365.0
    t_days = simulation["t_days"]
    release_rate = simulation["m_total"]
    valid = np.isfinite(t_days) & np.isfinite(release_rate)
    t_valid, rate_valid = t_days[valid], release_rate[valid]
    within_window = t_valid < analysis_days
    t_auc = np.append(t_valid[within_window], analysis_days)
    rate_auc = np.append(rate_valid[within_window], np.interp(analysis_days, t_valid, rate_valid))
    auc_mcg = np.trapezoid(rate_auc, t_auc)
    average_rate_mcg_day = auc_mcg / analysis_days
    medium_volume_ml = 100.0
    end_of_day_concentration_mcg_ml = average_rate_mcg_day / medium_volume_ml
    # With complete daily replacement and constant release over one day, the
    # medium concentration rises linearly from zero to C_end. Its time-average
    # is therefore C_end / 2, rather than the end-of-day sample concentration.
    average_concentration_mcg_ml = end_of_day_concentration_mcg_ml / 2.0
    water_solubility_mcg_ml = Cs_a * 1e6
    average_sink_ratio = average_concentration_mcg_ml / water_solubility_mcg_ml

    fig_medium = plt.figure(figsize=(15, 11), facecolor="white")
    ax_medium = fig_medium.add_axes([0, 0, 1, 1])
    ax_medium.axis("off")
    fig_medium.text(0.05, 0.91, "Appendix C. Estimated medium concentration: 0-5 years",
                    fontsize=18, fontweight="bold", va="top")
    fig_medium.text(0.05, 0.855,
                    "The calculation uses the fitted total-release profile in Figure 1E through year 5. "
                    "The fit itself uses 3.5 years of experimental data.", fontsize=10, va="top")
    lines = [
        ("1. Area under the fitted release-rate curve", 13, "bold"),
        (r"$AUC_{0-5y}=\int_0^{1825\ \mathrm{days}}m_{total}(t)\,dt$", 13, "normal"),
        (f"= {auc_mcg:,.1f} mcg = {auc_mcg / 1000.0:.3f} mg", 12, "normal"),
        ("", 8, "normal"),
        ("2. Average release rate over 5 years", 13, "bold"),
        (r"$\overline{m}=AUC_{0-5y}/1825\ \mathrm{days}$", 13, "normal"),
        (f"= {average_rate_mcg_day:.3f} mcg/day", 12, "normal"),
        ("", 8, "normal"),
        ("3. End-of-day concentration for 100 mL medium replaced daily", 13, "bold"),
        (r"$\overline{M}_{daily}=\overline{m}\times1\ \mathrm{day}$", 13, "normal"),
        (f"= {average_rate_mcg_day:.3f} mcg per daily medium sample", 12, "normal"),
        (r"$C_{end}=\overline{M}_{daily}/100\ \mathrm{mL}$", 13, "normal"),
        (f"= {end_of_day_concentration_mcg_ml:.4f} mcg/mL", 12, "normal"),
        ("", 8, "normal"),
        ("4. Time-averaged daily medium concentration", 13, "bold"),
        (r"$\overline{C}_{medium}=\dfrac{1}{1\ \mathrm{day}}\int_0^{1\ \mathrm{day}}\dfrac{\overline{m}\,t}{100\ \mathrm{mL}}dt=C_{end}/2$", 12, "normal"),
        (f"= {average_concentration_mcg_ml:.4f} mcg/mL", 12, "normal"),
    ]
    y_position = 0.77
    for line, fontsize, weight in lines:
        if line:
            fig_medium.text(0.08, y_position, line, fontsize=fontsize, fontweight=weight, va="top")
        y_position -= 0.035 if line else 0.012
    fig_medium.text(0.08, 0.165, "Relation to the sink-condition definition", fontsize=13,
                    fontweight="bold", va="top")
    fig_medium.text(0.08, 0.125,
                    f"Water solubility, C_s,a = {water_solubility_mcg_ml:.3f} mcg/mL.  "
                    f"Average sink ratio = C_medium/C_s,a = {average_sink_ratio:.4f} ({average_sink_ratio * 100:.2f}%).",
                    fontsize=11, va="top")
    fig_medium.text(0.08, 0.068,
                    f"The model uses a sink condition of {sink_condition:.3f} ({sink_condition * 100:.1f}%). "
                    "The corrected value is the time average over each daily replacement interval, assuming constant release.",
                    fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f6f6", edgecolor="#bdbdbd"))
    return fig_medium


audit_pages = create_model_audit_pages(sim_full)
medium_concentration_page = create_medium_concentration_appendix_page(sim_full)
audit_pages.append(medium_concentration_page)

# Report title page: placed first so each archived PDF is identifiable when
# viewed outside its versioned reports folder.
fig_title = plt.figure(figsize=(15, 11), facecolor="white")
title_ax = fig_title.add_axes([0, 0, 1, 1])
title_ax.axis("off")
fig_title.text(0.5, 0.61, REPORT_TITLE, ha="center", va="center",
               fontsize=25, fontweight="bold")
fig_title.text(0.5, 0.52, "PSS model fitting report", ha="center", va="center", fontsize=15)
fig_title.text(0.5, 0.45,
               "Organon release data - fitted DOS, PDL, ha_skin, and ha_ends",
               ha="center", va="center", fontsize=12)

# Save single PDF report containing all fitting and duration-sensitivity results
pdf_path = REPORTS_DIRECTORY / f"{REPORT_TITLE}.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig_title)
    pdf.savefig(fig)
    pdf.savefig(fig_zoom)
    pdf.savefig(fig2)
    run_data_duration_sensitivity(RESULTS_DIRECTORY, t_exp, m_exp, m_exp_sd_full=m_exp_sd, pdf=pdf)
    pdf.savefig(fig8)
    pdf.savefig(fig8_1)
    pdf.savefig(fig9)
    pdf.savefig(fig9_1)
    pdf.savefig(fig10)
    pdf.savefig(fig11)
    pdf.savefig(fig12)
    pdf.savefig(fig13)
    pdf.savefig(fig14)
    pdf.savefig(fig15)
    pdf.savefig(fig16)
    pdf.savefig(fig16_1)
    for audit_page in audit_pages:
        pdf.savefig(audit_page)

plt.close(fig8)
plt.close(fig8_1)
plt.close(fig9)
plt.close(fig9_1)
plt.close(fig10)
plt.close(fig11)
plt.close(fig12)
plt.close(fig13)
plt.close(fig14)
plt.close(fig15)
plt.close(fig16)
plt.close(fig16_1)
plt.close(fig_title)
for audit_page in audit_pages:
    plt.close(audit_page)

print(f"\nSingle PDF report saved: {pdf_path}")

if not args.no_show:
    plt.show()
