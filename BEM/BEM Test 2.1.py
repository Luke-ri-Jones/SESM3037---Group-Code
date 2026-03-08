import math
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
================================================================================
UPDATE LOG
================================================================================

v2.1 (current) - Luke Jones

  FIX 1 : Turns out we were using the last row of the polar data as the stall
           point, which is actually deep in post-stall territory (20 deg, not
           the real CL peak at 10 deg). That was messing up the Viterna
           constants and giving us wrong CL/CD whenever we extrapolated. Fixed
           it to find the actual CL peak with argmax instead.

  FIX 2 : We had no extrapolation at all for negative AoA below -10 deg —
           it just silently clamped to the edge value, which is wrong. Added
           the same Viterna treatment for the negative side, mirroring what
           we do for high positive AoA. Matters most at the root during startup.

  FIX 3 : AR was hardcoded as a default argument in polar_lookup, so if we
           ever run this on a different blade we'd have to remember to change
           the source. Moved it to a proper --ar CLI flag so it's obvious.

  FIX 4 : The convergence counter was broken — it was counting every single
           inner iteration step rather than just the elements that actually
           failed to converge, so the numbers were way inflated. Switched to
           a for...else so it only ticks up when an element genuinely runs
           out of iterations.

  FIX 5 : The Betz limit check was throwing a hard exception, which meant
           the whole sweep would crash if one point nudged slightly over 0.593
           due to noise. Changed it to a warning print so we still get told
           about it but don't lose all the results.

v2.0 - Luke Ryan

  - Added Convergence counter that prints out after each success
    (Can create a file instead if print is too annoying)
  - Changed def polar_lookup to include a Viterna-Corrigan extrapolation during startup/stalling
    (Change AR value depending on turbine blade geometry)
    
v1.0 - Luke Jones

  - Initial BEM implementation with Prandtl tip/hub loss and Buhl high-thrust
    correction.

================================================================================
0) What this code does
================================================================================

Wind-turbine blades are aerodynamic wings. At each radial station along the
blade, the local section experiences a "relative wind" whose direction and
magnitude depend on four quantities:

  • The free-stream wind speed (V0)              [m/s]
  • The rotor angular velocity (Omega)           [rad/s]
  • The axial induction factor (a)               [-]  — how much the rotor
        slows the wind upstream of the disc
  • The tangential induction factor (a')         [-]  — how much the rotor
        imparts swirl to the wake

Blade Element Momentum (BEM) theory couples two physical models:

  A) Blade Element (2-D airfoil) theory
     ─ Divide the blade into N thin annular strips.
     ─ For each strip, compute the local inflow angle (phi) and angle of
       attack (alpha) from the velocity triangle.
     ─ Look up the airfoil polar data CL(alpha), CD(alpha).
     ─ Resolve lift and drag into rotor-normal (thrust) and rotor-tangential
       (torque) force components.

  B) Actuator-Disc Momentum theory
     ─ The rotor is modelled as a permeable disc that extracts axial momentum
       and adds angular momentum to the flow.
     ─ For each annular ring, momentum conservation gives expressions for
       a and a' in terms of the blade-element forces.

  Because a and a' determine the velocity triangle which determines phi and
  alpha which determine the forces which determine a and a', the solution is
  found iteratively for each element:

      initialise (a, a') → compute velocity triangle → compute phi → compute
      alpha → look up CL, CD → compute new (a, a') → repeat until converged.

  Corrections applied on top of classical BEM:
    • Prandtl tip- and hub-loss factor (F)  — reduces loading at blade ends
    • Buhl empirical correction             — stabilises solution when a > 0.4
      (heavily loaded rotor, classical momentum theory breaks down)
    • Viterna-Corrigan extrapolation        — provides physically consistent
      CL and CD for angles of attack outside the measured polar range

================================================================================
'''


# =============================================================================
# Section 1 — CSV I/O helpers
# =============================================================================

def find_col(df, *must_contain):
    """
    Locate a DataFrame column by matching substrings in its header text.

    This allows flexible, case-insensitive column identification without
    requiring exact header strings, which is useful when CSV files from
    different sources use slightly different naming conventions.

    Parameters
    ----------
    df           : pd.DataFrame
        DataFrame whose column names are searched.
    *must_contain : str
        One or more substrings that must all appear (case-insensitive) in the
        target column name.

    Returns
    -------
    str
        The original (un-normalised) column name from df.

    Raises
    ------
    KeyError
        If no column matches all required substrings.
    """
    # Build a mapping of lowercase-stripped name -> original name
    cols = {c.lower().strip(): c for c in df.columns}

    for low, orig in cols.items():
        if all(s in low for s in must_contain):
            return orig

    raise KeyError(
        f"Could not find a column containing all of {must_contain}. "
        f"Available columns: {list(df.columns)}"
    )


def load_geometry(path: str) -> dict:
    """
    Read blade geometry from a CSV file and return it as a structured dict.

    The CSV must contain at minimum three columns identifiable by the
    substrings 'r' + '(m)' (radius), 'chord', and 'twist'. Column names are
    matched case-insensitively via find_col().

    Parameters
    ----------
    path : str
        Filesystem path to the geometry CSV.

    Returns
    -------
    dict with keys:
        r_nodes    : np.ndarray  — radial stations [m]
        c_nodes    : np.ndarray  — chord length at each station [m]
        beta_nodes : np.ndarray  — geometric twist at each station [deg]
        Rhub       : float       — innermost (hub) radius [m]
        Rtip       : float       — outermost (tip) radius [m]

    Raises
    ------
    ValueError
        If the radial stations are not strictly increasing after sorting.
    """
    df = pd.read_csv(path)

    # Identify columns flexibly from partial header matches
    r_col = find_col(df, "r", "(m)")
    c_col = find_col(df, "chord")
    b_col = find_col(df, "twist")

    r    = df[r_col].to_numpy(float)    # radial stations [m]
    c    = df[c_col].to_numpy(float)    # chord lengths   [m]
    beta = df[b_col].to_numpy(float)    # twist angles    [deg]

    # Ensure data is ordered from hub to tip (required for np.interp)
    idx = np.argsort(r)
    r, c, beta = r[idx], c[idx], beta[idx]

    if np.any(np.diff(r) <= 0):
        raise ValueError(
            "Blade geometry radii must be strictly increasing after sorting. "
            "Check for duplicate or out-of-order rows in the geometry CSV."
        )

    return {
        "r_nodes":    r,
        "c_nodes":    c,
        "beta_nodes": beta,
        "Rhub":       float(r[0]),    # hub radius [m]
        "Rtip":       float(r[-1]),   # tip radius [m]
    }


def load_polar(path: str) -> dict:
    """
    Read airfoil polar data (alpha -> CL, CD) from a CSV or TSV file.

    The function attempts tab-separated parsing first, then falls back to
    comma-separated. It also handles the edge case where the entire file is
    read as a single column (tab characters embedded in values).

    Parameters
    ----------
    path : str
        Filesystem path to the polar file.

    Returns
    -------
    dict with keys:
        alpha : np.ndarray  — angle of attack grid, sorted ascending [deg]
        cl    : np.ndarray  — lift coefficient at each alpha [-]
        cd    : np.ndarray  — drag coefficient at each alpha [-]

    Raises
    ------
    KeyError
        If the required columns (degree, cl, cd) cannot be found.
    """
    p = Path(path)

    # Attempt tab-separated parse; fall back to comma-separated on failure
    try:
        df = pd.read_csv(p, sep=r"\t", engine="python")
    except Exception:
        df = pd.read_csv(p)

    # Handle case where file is read as a single column with embedded tabs
    if df.shape[1] == 1:
        col   = df.columns[0]
        split = df[col].astype(str).str.split(r"\t", expand=True)
        split.columns = ["degree", "CL", "CD"]
        df    = split

    # Normalise all column names to lowercase for reliable access
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"degree", "cl", "cd"}
    if not required.issubset(df.columns):
        raise KeyError(
            f"Polar file must contain columns: {required}. "
            f"Found: {list(df.columns)}"
        )

    alpha = df["degree"].to_numpy(float)
    cl    = df["cl"].to_numpy(float)
    cd    = df["cd"].to_numpy(float)

    # Sort by alpha so interpolation routines behave correctly
    idx = np.argsort(alpha)
    return {"alpha": alpha[idx], "cl": cl[idx], "cd": cd[idx]}


def polar_lookup(polar: dict, alpha_deg: np.ndarray, AR: float = 15.0):
    """
    Return CL and CD at the requested angles of attack.

    Within the measured polar range the function uses linear interpolation.
    Outside the range it applies the Viterna-Corrigan (1982) flat-plate model
    to extrapolate physically consistent values up to ±90°.

    The stall reference point is taken at the CL peak (argmax), not the last
    data row. Using the last row — which may lie deep in post-stall — would
    produce incorrect Viterna constants and therefore wrong extrapolated values
    across the entire high-AoA regime.

    Both positive and negative extrapolation are handled symmetrically: for
    alpha below the data range, the polar is reflected about alpha = 0 and
    the same Viterna model is applied, then the CL sign is negated.

    Parameters
    ----------
    polar     : dict
        Polar dictionary from load_polar() with keys 'alpha', 'cl', 'cd'.
    alpha_deg : np.ndarray
        Requested angles of attack [deg]. May contain values outside the
        measured polar range.
    AR : float
        Blade aspect ratio used to estimate CD_max at 90° via the
        Viterna-Corrigan correlation: CD_max = 1.11 + 0.018 * AR.
        Default 15 is appropriate for the RISO 500 kW reference blade.

    Returns
    -------
    cl_out : np.ndarray  — interpolated / extrapolated lift coefficients [-]
    cd_out : np.ndarray  — interpolated / extrapolated drag coefficients [-]
    """
    a  = polar["alpha"]    # measured AoA grid [deg]
    cl = polar["cl"]       # measured CL values [-]
    cd = polar["cd"]       # measured CD values [-]

    # ------------------------------------------------------------------
    # Identify the true aerodynamic stall point as the CL peak.
    # The Viterna model requires conditions at stall onset, not at an
    # arbitrary point deep in the post-stall region.
    # ------------------------------------------------------------------
    stall_idx   = int(np.argmax(cl))
    alpha_s     = a[stall_idx]              # stall angle [deg]
    cl_s        = cl[stall_idx]             # CL at stall [-]
    cd_s        = cd[stall_idx]             # CD at stall [-]
    alpha_s_rad = np.radians(alpha_s)       # stall angle [rad]

    # Maximum drag coefficient at 90° — Viterna-Corrigan empirical relation
    cd_max = 1.11 + 0.018 * AR             # [-]

    # ------------------------------------------------------------------
    # Viterna-Corrigan constants for the HIGH-alpha (positive) regime.
    # Reference: Viterna & Corrigan (1982), NASA TM-82944.
    #
    # Drag:  CD(alpha) = B1 * sin²(alpha) + B2 * cos(alpha)
    # Lift:  CL(alpha) = A1 * sin(2*alpha) + A2 * cos²(alpha) / sin(alpha)
    # ------------------------------------------------------------------
    B1 = cd_max
    B2 = (cd_s - cd_max * np.sin(alpha_s_rad) ** 2) / np.cos(alpha_s_rad)
    A1 = B1 / 2.0
    A2 = (
        (cl_s - cd_max * np.sin(alpha_s_rad) * np.cos(alpha_s_rad))
        * np.sin(alpha_s_rad)
        / np.cos(alpha_s_rad) ** 2
    )

    # ------------------------------------------------------------------
    # Negative-stall reference (for extrapolation below the data range).
    # The negative stall point is taken as the CL minimum.
    # ------------------------------------------------------------------
    neg_stall_idx   = int(np.argmin(cl))
    alpha_ns        = a[neg_stall_idx]           # negative stall angle [deg]
    cl_ns           = cl[neg_stall_idx]          # CL at negative stall [-]
    cd_ns           = cd[neg_stall_idx]          # CD at negative stall [-]
    alpha_ns_rad    = np.radians(abs(alpha_ns))  # magnitude used in formula [rad]

    # Viterna constants for the negative-alpha extrapolation.
    # Symmetric flat-plate physics: treat |alpha| as the working angle,
    # compute positive CL, then negate at the end.
    B1_neg = cd_max
    B2_neg = (cd_ns - cd_max * np.sin(alpha_ns_rad) ** 2) / np.cos(alpha_ns_rad)
    A1_neg = B1_neg / 2.0
    A2_neg = (
        (abs(cl_ns) - cd_max * np.sin(alpha_ns_rad) * np.cos(alpha_ns_rad))
        * np.sin(alpha_ns_rad)
        / np.cos(alpha_ns_rad) ** 2
    )

    # Boundaries of the measured polar data
    alpha_min = a[0]     # lower data boundary [deg]
    alpha_max = a[-1]    # upper data boundary [deg]

    # ------------------------------------------------------------------
    # Evaluate CL and CD for each requested alpha.
    # Three regimes:
    #   1) alpha < alpha_min  →  negative Viterna extrapolation
    #   2) alpha_min <= alpha <= alpha_max  →  linear interpolation
    #   3) alpha > alpha_max  →  positive Viterna extrapolation
    # ------------------------------------------------------------------
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    cl_out    = np.zeros_like(alpha_deg)
    cd_out    = np.zeros_like(alpha_deg)

    for i, alpha in enumerate(alpha_deg):

        if alpha_min <= alpha <= alpha_max:
            # ---- Regime 2: interpolate within measured data ----
            cl_out[i] = np.interp(alpha, a, cl)
            cd_out[i] = np.interp(alpha, a, cd)

        elif alpha > alpha_max:
            # ---- Regime 3: Viterna-Corrigan (positive high AoA) ----
            # Clamp to 90° to keep trigonometric functions well-behaved
            a_rad      = np.radians(min(alpha, 90.0))
            cd_out[i]  = B1 * np.sin(a_rad) ** 2 + B2 * np.cos(a_rad)
            cl_out[i]  = A1 * np.sin(2.0 * a_rad) + A2 * np.cos(a_rad) ** 2 / np.sin(a_rad)

        else:
            # ---- Regime 1: Viterna-Corrigan (negative high AoA) ----
            # Work with the magnitude of alpha; negate CL at the end
            # to preserve the antisymmetric nature of lift.
            a_rad      = np.radians(min(abs(alpha), 90.0))
            cd_out[i]  = B1_neg * np.sin(a_rad) ** 2 + B2_neg * np.cos(a_rad)
            cl_mag     = A1_neg * np.sin(2.0 * a_rad) + A2_neg * np.cos(a_rad) ** 2 / np.sin(a_rad)
            cl_out[i]  = -cl_mag    # negate: negative AoA produces negative CL

    return cl_out, cd_out


# =============================================================================
# Section 2 — Geometry discretisation
# =============================================================================

def resample_elements(geom: dict, n: int):
    """
    Convert node-based blade geometry into element-based geometry.

    The blade span (hub to tip) is divided into n equal-width annular
    elements. Chord and twist at each element midpoint are obtained by
    linear interpolation from the input node arrays.

    Parameters
    ----------
    geom : dict
        Geometry dictionary from load_geometry().
    n    : int
        Number of annular blade elements.

    Returns
    -------
    r     : np.ndarray  — element midpoint radii [m],  shape (n,)
    dr    : np.ndarray  — element radial widths  [m],  shape (n,)
    chord : np.ndarray  — chord at each midpoint [m],  shape (n,)
    beta  : np.ndarray  — twist at each midpoint [deg],shape (n,)
    """
    # Uniformly spaced element boundaries from hub to tip
    r_edges = np.linspace(geom["Rhub"], geom["Rtip"], n + 1)   # [m]
    r       = 0.5 * (r_edges[:-1] + r_edges[1:])               # midpoints [m]
    dr      = r_edges[1:] - r_edges[:-1]                        # widths    [m]

    # Interpolate geometry onto element midpoints
    chord = np.interp(r, geom["r_nodes"], geom["c_nodes"])      # [m]
    beta  = np.interp(r, geom["r_nodes"], geom["beta_nodes"])   # [deg]

    return r, dr, chord, beta


# =============================================================================
# Section 3 — Prandtl tip and hub loss factor
# =============================================================================

def prandtl_F(
    B: int,
    r: float,
    Rhub: float,
    Rtip: float,
    phi: float,
    use_tip: bool = True,
    use_hub: bool = True,
    Fmin: float = 1e-4,
) -> float:
    """
    Compute the combined Prandtl tip- and hub-loss correction factor F.

    Prandtl's vortex model shows that a rotor with a finite number of blades
    experiences reduced loading near the tip and hub compared to an idealised
    actuator disc. The loss factor F in (0, 1] multiplies the effective momentum
    exchange and is applied to both the thrust and torque equations.

    Parameters
    ----------
    B        : int    — number of rotor blades [-]
    r        : float  — local element radius [m]
    Rhub     : float  — hub radius [m]
    Rtip     : float  — tip radius [m]
    phi      : float  — local inflow angle [rad]
    use_tip  : bool   — apply tip-loss correction
    use_hub  : bool   — apply hub-loss correction
    Fmin     : float  — minimum permitted value of F (prevents divide-by-zero)

    Returns
    -------
    float : combined loss factor F, clamped to [Fmin, 1]
    """
    sinp = max(abs(math.sin(phi)), 1e-8)    # |sin(phi)|; guarded against zero

    F = 1.0     # initialise combined factor as unity (no loss)

    if use_tip:
        f_tip = max((B / 2.0) * (Rtip - r) / (r * sinp), 0.0)
        F    *= (2.0 / math.pi) * math.acos(math.exp(-f_tip))

    if use_hub:
        f_hub = max((B / 2.0) * (r - Rhub) / (r * sinp), 0.0)
        F    *= (2.0 / math.pi) * math.acos(math.exp(-f_hub))

    return max(Fmin, min(1.0, F))


# =============================================================================
# Section 4 — Buhl high-thrust empirical correction
# =============================================================================

def buhl_a_from_CT(CT: float, F: float) -> float:
    """
    Derive the axial induction factor a from the annulus thrust coefficient.

    Classical momentum theory (CT = 4Fa(1-a)) breaks down for a > ~0.4 because
    the wake velocity becomes negative, which is unphysical. The Buhl (2005)
    empirical correction blends smoothly into a modified expression for heavily
    loaded conditions.

    Reference: Buhl, M.L. (2005), NREL/TP-500-36834.

    Parameters
    ----------
    CT : float — annulus thrust coefficient (blade-element estimate) [-]
    F  : float — Prandtl loss factor for this element [-]

    Returns
    -------
    float : axial induction factor a, clamped to [0, 0.95]
    """
    F  = max(F,  1e-6)    # guard against zero denominator
    CT = max(CT, 0.0)     # thrust coefficient must be non-negative

    CT_switch = 0.96 * F  # threshold between momentum and empirical regime

    if CT < CT_switch:
        # Standard momentum region: invert CT = 4 F a (1 - a)
        discriminant = max(0.0, 1.0 - CT / F)
        a = 0.5 * (1.0 - math.sqrt(discriminant))
    else:
        # High-thrust (turbulent-wake) region: Buhl empirical formula
        term  = CT * (50.0 - 36.0 * F) + 12.0 * F * (3.0 * F - 4.0)
        term  = max(0.0, term)
        denom = 36.0 * F - 50.0
        a     = (18.0 * F - 20.0 - 3.0 * math.sqrt(term)) / denom

    return max(0.0, min(0.95, a))


# =============================================================================
# Section 5 — BEM solver for a single operating point
# =============================================================================

def solve_bem_point(
    geom:      dict,
    polar:     dict,
    V0:        float,
    tsr:       float,
    pitch_deg: float = 0.0,
    n:         int   = 60,
    B:         int   = 3,
    rho:       float = 1.225,
    AR:        float = 15.0,
    use_tip:   bool  = True,
    use_hub:   bool  = True,
    use_buhl:  bool  = True,
    relax:     float = 0.25,
    tol:       float = 1e-6,
    itmax:     int   = 250,
):
    """
    Solve the BEM equations for one wind speed and return performance metrics.

    The blade is divided into n annular elements. Each element is solved
    independently by iterating on the induction factors (a, a') until the
    change per iteration falls below the convergence tolerance tol, or until
    itmax iterations are exhausted.

    Parameters
    ----------
    geom      : dict   — blade geometry from load_geometry()
    polar     : dict   — airfoil polar from load_polar()
    V0        : float  — free-stream wind speed [m/s]
    tsr       : float  — tip-speed ratio Omega*R / V0 [-]
    pitch_deg : float  — collective blade pitch angle [deg]
                         (positive pitch reduces angle of attack)
    n         : int    — number of annular blade elements [-]
    B         : int    — number of rotor blades [-]
    rho       : float  — air density [kg/m³]
    AR        : float  — blade aspect ratio for Viterna CD_max correlation [-]
    use_tip   : bool   — enable Prandtl tip-loss correction
    use_hub   : bool   — enable Prandtl hub-loss correction
    use_buhl  : bool   — enable Buhl high-thrust correction for axial induction
    relax     : float  — under-relaxation factor applied to induction updates
                         (0 < relax <= 1; smaller values improve stability)
    tol       : float  — convergence criterion: |delta_a| and |delta_a'| < tol
    itmax     : int    — maximum iterations per element before declaring failure

    Returns
    -------
    summary : dict
        Scalar performance quantities:
          V0_mps, rpm, P_kW, Q_Nm, T_N, CP, CT
    dist : pd.DataFrame
        Spanwise distributions of all computed quantities.
    """
    R     = geom["Rtip"]                            # rotor tip radius [m]
    omega = tsr * V0 / R                            # angular velocity [rad/s]
    rpm   = omega * 60.0 / (2.0 * math.pi)          # rotor speed [rev/min]

    # Discretise blade geometry onto n element midpoints
    r, dr, chord, beta = resample_elements(geom, n)

    # Initialise induction factor arrays.
    # a = 0.30 is a physically reasonable first guess for a wind turbine.
    a  = np.full(n, 0.30)    # axial induction factor [-]
    ap = np.zeros(n)         # tangential induction factor [-]

    # Output arrays filled during the element loop
    phi   = np.zeros(n)      # converged inflow angle [rad]
    alpha = np.zeros(n)      # converged angle of attack [deg]
    F     = np.ones(n)       # Prandtl loss factor [-]
    CL    = np.zeros(n)      # lift coefficient [-]
    CD    = np.zeros(n)      # drag coefficient [-]
    Fn    = np.zeros(n)      # normal (thrust) force per unit span [N/m]
    Ft    = np.zeros(n)      # tangential (torque) force per unit span [N/m]

    n_fail = 0               # count of elements that did not converge

    # ------------------------------------------------------------------
    # Element-by-element iterative solve
    # Each annular ring is aerodynamically independent (BEM assumption).
    # ------------------------------------------------------------------
    for i in range(n):

        # Local blade solidity: fraction of annulus area covered by blades
        sigma = (B * chord[i]) / (2.0 * math.pi * r[i])    # [-]

        ai  = float(a[i])     # working axial induction guess [-]
        api = float(ap[i])    # working tangential induction guess [-]

        # Inner iteration: converge (a, a') for element i
        # The for...else construct means the else block executes only if the
        # loop completes without hitting the 'break' (i.e. did not converge).
        for _ in range(itmax):

            # Velocity triangle
            Vax  = V0 * (1.0 - ai)                  # axial velocity component  [m/s]
            Vtan = max(omega * r[i] * (1.0 + api), 1e-12)  # tangential component [m/s]

            # Inflow angle: angle between rotor plane and relative velocity vector
            ph = math.atan2(Vax, Vtan)               # [rad]
            ph = max(1e-6, min(ph, math.pi / 2.0 - 1e-6))  # guard singularities

            # Prandtl tip/hub loss factor for this element
            Fi = prandtl_F(
                B=B, r=float(r[i]),
                Rhub=geom["Rhub"], Rtip=geom["Rtip"],
                phi=ph, use_tip=use_tip, use_hub=use_hub,
            )

            # Angle of attack: inflow angle minus local blade pitch angle
            # (geometric twist + collective pitch together define the local
            # pitch angle of the chord line relative to the rotor plane)
            aoi = math.degrees(ph) - (beta[i] + pitch_deg)  # [deg]

            # Aerodynamic coefficients from polar (with Viterna extrapolation)
            cli, cdi = polar_lookup(polar, np.array([aoi]), AR=AR)
            cli, cdi = float(cli[0]), float(cdi[0])

            # Force coefficients in rotor-aligned frame:
            #   Cn (normal, i.e. thrust direction)
            #   Ct (tangential, i.e. torque direction)
            Cn = cli * math.cos(ph) + cdi * math.sin(ph)    # [-]
            Ct = cli * math.sin(ph) - cdi * math.cos(ph)    # [-]

            # Relative dynamic pressure factor: W² = Vax² + Vtan²
            W2  = Vax ** 2 + Vtan ** 2                      # [m²/s²]

            # Annulus thrust coefficient (blade-element estimate)
            CTi = sigma * Cn * W2 / max(V0 ** 2, 1e-12)    # [-]

            # --- Update axial induction factor ---
            if use_buhl:
                a_new = buhl_a_from_CT(CTi, Fi)
            else:
                # Classical momentum expression: CT = 4 F sin²(phi) * a / sigma / Cn
                sin2  = max(math.sin(ph) ** 2, 1e-10)
                a_new = 1.0 / (4.0 * Fi * sin2 / (sigma * max(Cn, 1e-12)) + 1.0)

            # --- Update tangential induction factor ---
            # From: CT_tang = 4 F sin(phi) cos(phi) * a' * (1 + a')
            if abs(Ct) < 1e-12:
                ap_new = api    # negligible tangential force; no update
            else:
                denom  = (4.0 * Fi * math.sin(ph) * math.cos(ph)) / (sigma * Ct) - 1.0
                ap_new = api if abs(denom) < 1e-12 else 1.0 / denom

            # Under-relaxation: blend new estimate with previous value to
            # prevent oscillation in the iteration
            ai_next  = (1.0 - relax) * ai  + relax * a_new
            api_next = (1.0 - relax) * api + relax * ap_new

            # Convergence check: both induction factors must change by less
            # than tol in this iteration
            if abs(ai_next - ai) < tol and abs(api_next - api) < tol:
                ai,  api  = ai_next, api_next
                phi[i]    = ph
                alpha[i]  = aoi
                F[i]      = Fi
                CL[i], CD[i] = cli, cdi
                break

            ai, api = ai_next, api_next

        else:
            # The for loop completed without a 'break' — element did not converge.
            # Store the last iterate and flag the failure for the summary report.
            phi[i]       = ph
            alpha[i]     = aoi
            F[i]         = Fi
            CL[i], CD[i] = cli, cdi
            n_fail      += 1

        # Store converged (or best-available) induction factors
        a[i], ap[i] = ai, api

        # Recompute final velocity triangle and forces at converged induction
        Vax  = V0 * (1.0 - ai)
        Vtan = omega * r[i] * (1.0 + api)
        W2   = Vax ** 2 + Vtan ** 2
        ph   = phi[i]

        Cn   = CL[i] * math.cos(ph) + CD[i] * math.sin(ph)
        Ct   = CL[i] * math.sin(ph) - CD[i] * math.cos(ph)

        # Blade forces per unit span [N/m]:  F = ½ rho W² c C
        Fn[i] = 0.5 * rho * W2 * chord[i] * Cn
        Ft[i] = 0.5 * rho * W2 * chord[i] * Ct

    # Report convergence outcome for this wind speed
    if n_fail == 0:
        print(f"  V0 = {V0:5.2f} m/s  — all {n} elements converged.")
    else:
        print(
            f"  V0 = {V0:5.2f} m/s  — WARNING: {n_fail}/{n} elements "
            f"did not converge within {itmax} iterations."
        )

    # ------------------------------------------------------------------
    # Integrate annular contributions to global performance metrics
    # ------------------------------------------------------------------
    dT = B * Fn * dr            # thrust contribution from each annulus [N]
    dQ = B * Ft * r * dr        # torque contribution from each annulus [N·m]

    T  = float(np.sum(dT))      # total rotor thrust [N]
    Q  = float(np.sum(dQ))      # total rotor torque [N·m]
    P  = float(omega * Q)       # total rotor power [W]

    A  = math.pi * R ** 2       # rotor swept area [m²]
    CP = P / (0.5 * rho * A * V0 ** 3)     # power coefficient [-]
    CT = T / (0.5 * rho * A * V0 ** 2)     # thrust coefficient [-]

    # ------------------------------------------------------------------
    # Assemble output structures
    # ------------------------------------------------------------------
    dist = pd.DataFrame({
        "r_m":        r,
        "dr_m":       dr,
        "chord_m":    chord,
        "twist_deg":  beta,
        "a":          a,
        "a_prime":    ap,
        "phi_deg":    np.degrees(phi),
        "alpha_deg":  alpha,
        "F":          F,
        "CL":         CL,
        "CD":         CD,
        "Fn_N_per_m": Fn,
        "Ft_N_per_m": Ft,
        "dT_N":       dT,
        "dQ_Nm":      dQ,
    })

    summary = {
        "V0_mps": float(V0),
        "rpm":    float(rpm),
        "P_kW":   float(P / 1000.0),
        "Q_Nm":   float(Q),
        "T_N":    float(T),
        "CP":     float(CP),
        "CT":     float(CT),
    }

    return summary, dist


# =============================================================================
# Section 6 — Wind-speed sweep (verification / power-curve generation)
# =============================================================================

def verify(
    geom:   dict,
    polar:  dict,
    winds:  list,
    tsr:    float,
    pitch:  float,
    n:      int,
    AR:     float,
    outdir: str,
    outNm:  str,
) -> pd.DataFrame:
    """
    Solve the BEM equations across a range of wind speeds and save results.

    For each wind speed in 'winds', solve_bem_point() is called at constant
    TSR and pitch. Results are aggregated into a summary table (one row per
    wind speed) and also saved as CSV files.

    Parameters
    ----------
    geom   : dict   — blade geometry from load_geometry()
    polar  : dict   — airfoil polar from load_polar()
    winds  : list   — wind speeds to evaluate [m/s]
    tsr    : float  — constant tip-speed ratio [-]
    pitch  : float  — collective pitch angle [deg]
    n      : int    — number of blade elements per solve [-]
    AR     : float  — blade aspect ratio for Viterna model [-]
    outdir : str    — output directory for saved CSV files
    outNm  : str    — filename for the summary CSV

    Returns
    -------
    pd.DataFrame : summary table with columns V0_mps, rpm, P_kW, Q_Nm, T_N, CP, CT
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows  = []      # per-wind-speed summary dicts
    dists = {}      # wind speed -> spanwise distribution DataFrame

    print("\nRunning BEM sweep...")

    for V in winds:
        s, dist = solve_bem_point(
            geom=geom,
            polar=polar,
            V0=float(V),
            tsr=float(tsr),
            pitch_deg=float(pitch),
            n=int(n),
            AR=float(AR),
        )
        rows.append(s)
        dists[float(V)] = dist

    # Build and sort summary table
    df = pd.DataFrame(rows).sort_values("V0_mps").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Sanity checks
    # A warning is issued rather than raising an exception so that the
    # full sweep result is still returned and saved even when individual
    # points are suspect.
    # ------------------------------------------------------------------
    betz_violations = df[df["CP"] > 0.593]
    if not betz_violations.empty:
        print(
            f"\n  WARNING: CP exceeded the Betz limit (0.593) at "
            f"{len(betz_violations)} wind speed(s): "
            f"{betz_violations['V0_mps'].tolist()}. "
            f"Check polar data and solver settings."
        )

    negative_power = df[df["P_kW"] < -1e-6]
    if not negative_power.empty:
        print(
            f"\n  WARNING: Negative power detected at "
            f"{len(negative_power)} wind speed(s): "
            f"{negative_power['V0_mps'].tolist()}. "
            f"This is unexpected for normal turbine operation."
        )

    # Save summary and a representative spanwise distribution to CSV
    df.to_csv(outdir / outNm, index=False)

    Vmid = sorted(dists.keys())[len(dists) // 2]
    dists[Vmid].to_csv(outdir / f"bem_distributions_V{Vmid:.1f}.csv", index=False)

    return df


# =============================================================================
# Section 7 — Command-line interface
# =============================================================================

def main():
    """
    Entry point for command-line execution.

    All solver parameters are exposed as optional CLI arguments so the code
    can be run on different turbines or operating conditions without modifying
    the source. Run with --help for a full description of each argument.
    """
    ap = argparse.ArgumentParser(
        description="Blade Element Momentum (BEM) solver for horizontal-axis wind turbines."
    )

    ap.add_argument(
        "--geom", type=str,
        default="RISO-A1-A18 Profile for 500kW Reference Turbine Blade.csv",
        help="Path to blade geometry CSV (must contain r (m), chord, and twist columns).",
    )
    ap.add_argument(
        "--polar", type=str,
        default="RISO-A1-A18-Lift-Drag-Characteristics.csv",
        help="Path to airfoil polar CSV (must contain degree, CL, and CD columns).",
    )
    ap.add_argument(
        "--winds", type=float, nargs=3,
        default=[4, 20, 0.25],
        metavar=("V_MIN", "V_MAX", "V_STEP"),
        help="Wind-speed sweep defined as: V_MIN V_MAX V_STEP (all inclusive). "
             "Example: --winds 4 16 1",
    )
    ap.add_argument(
        "--tsr", type=float, default=7.5,
        help="Constant tip-speed ratio (Omega * R / V0) applied across all wind speeds [-].",
    )
    ap.add_argument(
        "--pitch", type=float, default=0.0,
        help="Collective blade pitch angle [deg]. Positive pitch reduces angle of attack.",
    )
    ap.add_argument(
        "--n", type=int, default=60,
        help="Number of annular blade elements used in the BEM discretisation.",
    )
    ap.add_argument(
        "--ar", type=float, default=15.0,
        help="Blade aspect ratio used in the Viterna-Corrigan CD_max correlation "
             "(CD_max = 1.11 + 0.018 * AR). Default 15 suits the RISO 500 kW blade.",
    )
    ap.add_argument(
        "--outdir", type=str, default="bem_outputs",
        help="Directory in which output CSV files and plots are saved.",
    )
    ap.add_argument(
        "--outNm", type=str, default="bem_summary.csv",
        help="Filename for the summary CSV output.",
    )

    args = ap.parse_args()

    # Validate and expand wind-speed range
    Vmin, Vmax, Vstep = args.winds
    if Vstep <= 0:
        raise ValueError("--winds step must be greater than zero.")
    if Vmax < Vmin:
        raise ValueError("--winds V_MAX must be greater than or equal to V_MIN.")

    # Use 0.5 * step tolerance in the upper bound to ensure Vmax is included
    # despite floating-point rounding in np.arange
    winds = np.arange(Vmin, Vmax + 0.5 * Vstep, Vstep).tolist()

    # Load input data
    geom  = load_geometry(args.geom)
    polar = load_polar(args.polar)

    # Run wind-speed sweep
    df = verify(
        geom=geom,
        polar=polar,
        winds=winds,
        tsr=args.tsr,
        pitch=args.pitch,
        n=args.n,
        AR=args.ar,
        outdir=args.outdir,
        outNm=args.outNm,
    )

    # Print formatted summary table to console
    print("\nVerification summary (constant TSR):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved summary CSV  : {Path(args.outdir) / args.outNm}")


    # =========================================================================
    # Section 8 — Diagnostic plots
    # =========================================================================

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.ravel()
    x   = df["V0_mps"]

    # Panel 1: Rotor speed vs wind speed
    axs[0].plot(x, df["rpm"], marker="o", linewidth=1.5)
    axs[0].set_ylabel("Rotor speed [rpm]")
    axs[0].set_title("Rotor Speed")
    axs[0].grid(True)

    # Panel 2: Electrical power vs wind speed
    axs[1].plot(x, df["P_kW"], marker="o", linewidth=1.5, color="tab:orange")
    axs[1].set_ylabel("Power [kW]")
    axs[1].set_title("Power")
    axs[1].grid(True)

    # Panel 3: Aerodynamic torque vs wind speed
    axs[2].plot(x, df["Q_Nm"], marker="o", linewidth=1.5, color="tab:green")
    axs[2].set_xlabel("Free-stream wind speed V0 [m/s]")
    axs[2].set_ylabel("Torque [N·m]")
    axs[2].set_title("Torque")
    axs[2].grid(True)

    # Panel 4: Rotor thrust vs wind speed
    axs[3].plot(x, df["T_N"], marker="o", linewidth=1.5, color="tab:red")
    axs[3].set_xlabel("Free-stream wind speed V0 [m/s]")
    axs[3].set_ylabel("Thrust [N]")
    axs[3].set_title("Thrust")
    axs[3].grid(True)

    fig.suptitle(
        f"BEM performance sweep — constant TSR = {args.tsr}, pitch = {args.pitch}°",
        fontsize=13,
    )
    fig.tight_layout()

    plot_path = Path(args.outdir) / "bem_plots_4panel.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Saved diagnostic plots: {plot_path}")

    plt.show()


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    main()

