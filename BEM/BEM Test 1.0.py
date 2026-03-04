import math                     # math functions (sin, cos, atan2, sqrt, pi, etc.)
from pathlib import Path         # filesystem paths (portable)
import argparse                 # command-line arguments
import numpy as np              # arrays + fast maths
import pandas as pd             # CSV reading/writing + tables
import matplotlib.pyplot as plt  # plotting

'''
================================================================================
0) What this code does
================================================================================

Wind-turbine blades are essentially wings. At each radius along the blade, the
local section experiences a “relative wind” that depends on:
  • The free-stream wind speed (V∞)
  • How fast the rotor spins (Ω)
  • How much the rotor slows the wind down (axial induction, a)
  • How much the rotor adds swirl to the wake (tangential induction, a′)

BEM combines two ideas:

A) Blade Element (2D airfoil) theory:
   - Split the blade into many small radial “elements”.
   - For each element, compute the local inflow angle (φ) and angle of attack (α).
   - Use airfoil polar data (CL(α), CD(α)) to compute aerodynamic forces.

B) Momentum (actuator disc) theory:
   - The rotor extracts momentum from the wind. This is captured through
     induction factors a and a′.
   - For each annular ring, the forces predicted by the blade element model
     must be consistent with momentum theory.

Because a and a′ affect the inflow angle φ (and therefore α, CL, CD, forces),
we must solve each blade element iteratively:
   guess (a, a′) → compute φ → compute α → lookup CL/CD → compute new (a, a′)
   repeat until converged.

'''
# =============================================================================
# 1) CSV loading 
# =============================================================================
def find_col(df, *must_contain):
    """
    df           : pandas DataFrame containing CSV columns
    must_contain : strings that must appear inside the target column name (case-insensitive)

    Returns:
      The original column name from df whose lowercase version contains all required substrings.
    """
    cols = {c.lower().strip(): c for c in df.columns}  # mapping: lowercase_name -> original_name

    for low, orig in cols.items():                     # low = lowercase column name, orig = original
        if all(s in low for s in must_contain):        # check every required substring is present
            return orig

    raise KeyError(f"Could not find column containing {must_contain}. Columns: {list(df.columns)}")


def load_geometry(path: str):
    """
    Load blade geometry from CSV.

    path : filename/path to geometry CSV

    Returns a dict with:
      r_nodes : radius nodes [m]
      c_nodes : chord at nodes [m]
      beta_nodes : twist at nodes [deg]
      Rhub : first radius node (start of lifting surface) [m]
      Rtip : last radius node (rotor radius) [m]
    """
    df = pd.read_csv(path)                         # read CSV into a DataFrame

    # Identify columns robustly by searching header strings
    r_col = find_col(df, "r", "(m)")               # radius column (dataset uses something like "r (m)")
    c_col = find_col(df, "chord")                  # chord column
    b_col = find_col(df, "twist")                  # twist column

    # Convert to numpy arrays of floats
    r = df[r_col].to_numpy(float)                  # radii at input nodes [m]
    c = df[c_col].to_numpy(float)                  # chord at nodes [m]
    beta = df[b_col].to_numpy(float)               # twist at nodes [deg]

    # Sort by radius so arrays increase monotonically (important for interpolation)
    idx = np.argsort(r)                            # sorting indices
    r, c, beta = r[idx], c[idx], beta[idx]

    # Simple check: radii must be strictly increasing
    if np.any(np.diff(r) <= 0):
        raise ValueError("Radius must be strictly increasing.")

    return {
        "r_nodes": r,
        "c_nodes": c,
        "beta_nodes": beta,
        "Rhub": float(r[0]),                        # hub/root radius [m]
        "Rtip": float(r[-1]),                       # tip radius / rotor radius [m]
    }


def load_polar(path: str):
    """
    Load polar data (alpha -> CL, CD).

    path : filename/path to polar (often tab-separated for RISO dataset)

    Returns a dict with:
      alpha : angle of attack grid [deg]
      cl    : lift coefficient at each alpha [-]
      cd    : drag coefficient at each alpha [-]
    """
    p = Path(path)

    # Try TSV first (tab-separated), if that fails, fall back to standard CSV
    try:
        df = pd.read_csv(p, sep=r"\t", engine="python")
    except Exception:
        df = pd.read_csv(p)

    # Some files might come in as one big column; split by tabs if needed
    if df.shape[1] == 1:
        col = df.columns[0]
        split = df[col].astype(str).str.split(r"\t", expand=True)
        split.columns = ["degree", "CL", "CD"]      # standard expected names
        df = split

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure required columns exist
    if not {"degree", "cl", "cd"}.issubset(df.columns):
        raise KeyError(f"Polar needs columns degree, CL, CD. Found: {list(df.columns)}")

    # Extract arrays
    a = df["degree"].to_numpy(float)               # angle of attack grid [deg]
    cl = df["cl"].to_numpy(float)                  # lift coefficient [-]
    cd = df["cd"].to_numpy(float)                  # drag coefficient [-]

    # Sort by alpha to keep interpolation stable
    idx = np.argsort(a)
    return {"alpha": a[idx], "cl": cl[idx], "cd": cd[idx]}


def polar_lookup(polar, alpha_deg):
    """
    Interpolate polar data at desired alpha.

    polar     : dict with polar["alpha"], polar["cl"], polar["cd"]
    alpha_deg : numpy array of requested angles of attack [deg]

    Returns:
      cl_i, cd_i : interpolated CL and CD arrays (clamped to end values outside range)
    """
    a = polar["alpha"]                              # polar alpha array [deg]
    cl = polar["cl"]                                # polar CL array [-]
    cd = polar["cd"]                                # polar CD array [-]

    # np.interp clamps if we provide left/right values
    cl_i = np.interp(alpha_deg, a, cl, left=cl[0], right=cl[-1])
    cd_i = np.interp(alpha_deg, a, cd, left=cd[0], right=cd[-1])
    return cl_i, cd_i


# =============================================================================
# 2) Geometry discretisation: resample nodes to element midpoints
# =============================================================================
def resample_elements(geom, n):
    """
    Turn node-based geometry into element-based geometry using midpoints.

    geom : geometry dict from load_geometry()
    n    : number of annular elements (rings)

    Returns:
      r     : element midpoint radii [m]
      dr    : element widths [m]
      chord : chord at each element midpoint [m]
      beta  : twist at each element midpoint [deg]
    """
    # r_edges are element boundaries from hub to tip
    r_edges = np.linspace(geom["Rhub"], geom["Rtip"], n + 1)  # [m]
    r = 0.5 * (r_edges[:-1] + r_edges[1:])                   # midpoints [m]
    dr = r_edges[1:] - r_edges[:-1]                           # widths [m]

    # Interpolate chord and twist from nodes onto midpoints
    chord = np.interp(r, geom["r_nodes"], geom["c_nodes"])    # chord [m]
    beta = np.interp(r, geom["r_nodes"], geom["beta_nodes"])  # twist [deg]
    return r, dr, chord, beta


# =============================================================================
# 3) Prandtl tip/hub loss factor
# =============================================================================
def prandtl_F(B, r, Rhub, Rtip, phi, use_tip=True, use_hub=True, Fmin=1e-4):
    """
    Compute Prandtl loss factor F (reduces loading near tip/hub).

    B     : number of blades [-]
    r     : local radius [m]
    Rhub  : hub/root radius [m]
    Rtip  : tip radius [m]
    phi   : inflow angle [rad]
    use_tip, use_hub : whether to apply tip and/or hub loss
    Fmin  : minimum allowed F to prevent divide-by-zero issues

    Returns:
      F in (Fmin, 1]
    """
    sinp = abs(math.sin(phi))        # |sin(phi)| (appears in loss formula)
    sinp = max(sinp, 1e-8)           # prevent division by zero

    F = 1.0                          # combined loss factor (tip * hub)

    if use_tip:
        # Tip loss "f" parameter
        f = (B / 2) * (Rtip - r) / (r * sinp)
        f = max(f, 0.0)
        # Prandtl tip factor
        F_tip = (2 / math.pi) * math.acos(math.exp(-f))
        F *= F_tip

    if use_hub:
        # Hub loss "f" parameter
        f = (B / 2) * (r - Rhub) / (r * sinp)
        f = max(f, 0.0)
        # Prandtl hub factor
        F_hub = (2 / math.pi) * math.acos(math.exp(-f))
        F *= F_hub

    # Clamp to [Fmin, 1]
    return max(Fmin, min(1.0, F))


# =============================================================================
# 4) High-thrust correction (Buhl-style) for axial induction
# =============================================================================
def buhl_a_from_CT(CT, F):
    """
    Compute axial induction 'a' from annulus thrust coefficient CT using
    a common Buhl-style empirical correction for high loading.

    CT : annulus thrust coefficient (non-dimensional) [-]
    F  : Prandtl loss factor [-]

    Returns:
      a in [0, 0.95]
    """
    F = max(F, 1e-6)               # avoid divide-by-zero
    CT = max(0.0, CT)              # thrust coefficient should not be negative

    CTs = 0.96 * F                 # switching threshold (typical)

    if CT < CTs:
        # "normal" momentum region: CT = 4 F a (1-a)
        x = max(0.0, 1.0 - CT / F)
        a = 0.5 * (1.0 - math.sqrt(x))
    else:
        # high-thrust region: empirical expression
        term = CT * (50.0 - 36.0 * F) + 12.0 * F * (3.0 * F - 4.0)
        term = max(0.0, term)
        denom = (36.0 * F - 50.0)
        a = (18.0 * F - 20.0 - 3.0 * math.sqrt(term)) / denom

    return max(0.0, min(0.95, a))


# =============================================================================
# 5) Main solver for ONE operating point (one wind speed)
# =============================================================================
def solve_bem_point(
    geom, polar,
    V0,                  # free-stream wind speed [m/s]
    tsr,                 # tip-speed ratio lambda = Omega*R / V0 [-]
    pitch_deg=0.0,       # collective pitch angle [deg] (positive reduces AoA if twist is defined usual way)
    n=60,                # number of blade elements (annuli) [-]
    B=3,                 # number of blades [-]
    rho=1.225,           # air density [kg/m^3]
    use_tip=True,        # enable Prandtl tip loss
    use_hub=True,        # enable Prandtl hub loss
    use_buhl=True,       # enable high-thrust correction for a
    relax=0.25,          # under-relaxation factor for a and a' [-]
    tol=1e-6,            # convergence tolerance for a and a' [-]
    itmax=250            # max iterations per element [-]
):
    """
    Returns:
      summary : dict with V0, rpm, P_kW, Q_Nm, T_N, CP, CT
      dist    : DataFrame with spanwise distributions (r, chord, twist, a, a', phi, alpha, etc.)
    """
    R = geom["Rtip"]                          # rotor radius [m]

    # Omega from TSR definition: TSR = Omega*R / V0  => Omega = TSR*V0/R
    omega = tsr * V0 / R                      # angular speed [rad/s]
    rpm = omega * 60.0 / (2.0 * math.pi)      # rotor speed [rev/min]

    # Discretise geometry into annular elements
    r, dr, chord, beta = resample_elements(geom, n)

    # Induction arrays:
    a = np.full(n, 0.30)                      # axial induction factor a [-]
    ap = np.zeros(n)                          # tangential induction factor a' [-]

    # Arrays we will fill for outputs:
    phi = np.zeros(n)                         # inflow angle phi [rad]
    alpha = np.zeros(n)                       # angle of attack alpha [deg]
    F = np.ones(n)                            # Prandtl loss factor [-]
    CL = np.zeros(n)                          # lift coefficient [-]
    CD = np.zeros(n)                          # drag coefficient [-]
    Fn = np.zeros(n)                          # normal force per unit span [N/m]
    Ft = np.zeros(n)                          # tangential force per unit span [N/m]

    # Loop over each annulus (independent ring solve)
    for i in range(n):
        # Local solidity: sigma = B*c / (2*pi*r)
        sigma = (B * chord[i]) / (2 * math.pi * r[i])  # [-]

        # Start with current guesses for this element
        ai = float(a[i])                      # current axial induction guess [-]
        api = float(ap[i])                    # current tangential induction guess [-]

        # Iterate to solve (a, a') for this element
        for _ in range(itmax):
            # Local axial velocity at rotor plane:
            # Vax = V0*(1-a)
            Vax = V0 * (1 - ai)               # [m/s]

            # Local tangential velocity seen by blade:
            # Vtan = Omega*r*(1+a')
            Vtan = omega * r[i] * (1 + api)   # [m/s]
            Vtan = max(Vtan, 1e-12)           # avoid zero division / weird angles

            # Inflow angle phi between rotor plane and relative wind:
            # phi = atan(Vax / Vtan)
            ph = math.atan2(Vax, Vtan)        # [rad]
            # keep away from 0 and 90 deg to avoid sin/cos singularities
            ph = max(1e-6, min(ph, math.pi / 2 - 1e-6))

            # Prandtl losses at this element
            Fi = prandtl_F(
                B=B,
                r=float(r[i]),
                Rhub=geom["Rhub"],
                Rtip=geom["Rtip"],
                phi=ph,
                use_tip=use_tip,
                use_hub=use_hub
            )                                  # [-]

            # Angle of attack:
            # alpha = phi(deg) - (twist + pitch)
            aoi = math.degrees(ph) - (beta[i] + pitch_deg)  # [deg]

            # Polar lookup gives CL and CD for this alpha
            cli, cdi = polar_lookup(polar, np.array([aoi]))
            cli, cdi = float(cli[0]), float(cdi[0])         # scalar values

            # Resolve lift/drag into normal/tangential force coefficients:
            # Cn = CL*cos(phi) + CD*sin(phi)
            # Ct = CL*sin(phi) - CD*cos(phi)
            Cn = cli * math.cos(ph) + cdi * math.sin(ph)    # [-]
            Ct = cli * math.sin(ph) - cdi * math.cos(ph)    # [-]

            # Relative wind speed squared at the blade section:
            # W^2 = Vax^2 + Vtan^2
            W2 = Vax * Vax + Vtan * Vtan                     # [m^2/s^2]

            # Annulus thrust coefficient based on blade-element forces:
            # CTi ≈ sigma * Cn * (W^2 / V0^2)
            CTi = sigma * Cn * (W2 / max(V0 * V0, 1e-12))    # [-]

            # Update axial induction a:
            if use_buhl:
                a_new = buhl_a_from_CT(CTi, Fi)              # [-]
            else:
                # Classical momentum form (with loss factor in denominator)
                sin2 = max(math.sin(ph) ** 2, 1e-10)
                a_new = 1.0 / ((4.0 * Fi * sin2) / (sigma * max(Cn, 1e-12)) + 1.0)

            # Update tangential induction a':
            # ap = 1 / ( (4 F sin(phi) cos(phi))/(sigma Ct) - 1 )
            if abs(Ct) < 1e-12:
                ap_new = api                                 # if Ct ~ 0, keep current value
            else:
                denom = (4.0 * Fi * math.sin(ph) * math.cos(ph)) / (sigma * Ct) - 1.0
                ap_new = api if abs(denom) < 1e-12 else 1.0 / denom

            # Under-relaxation (stability):
            ai_next = (1 - relax) * ai + relax * a_new        # [-]
            api_next = (1 - relax) * api + relax * ap_new      # [-]

            # Convergence check:
            if abs(ai_next - ai) < tol and abs(api_next - api) < tol:
                ai, api = ai_next, api_next
                phi[i], alpha[i], F[i] = ph, aoi, Fi
                CL[i], CD[i] = cli, cdi
                break

            # Not converged yet: update and keep iterating
            ai, api = ai_next, api_next

        # Store converged induction factors
        a[i], ap[i] = ai, api

        # Once converged, compute forces per unit span:
        Vax = V0 * (1 - ai)                         # [m/s]
        Vtan = omega * r[i] * (1 + api)             # [m/s]
        W2 = Vax * Vax + Vtan * Vtan                # [m^2/s^2]

        ph = phi[i]                                  # [rad]
        Cn = CL[i] * math.cos(ph) + CD[i] * math.sin(ph)
        Ct = CL[i] * math.sin(ph) - CD[i] * math.cos(ph)

        # Normal and tangential forces per unit span:
        # F = 0.5*rho*W^2*c*C
        Fn[i] = 0.5 * rho * W2 * chord[i] * Cn       # [N/m]
        Ft[i] = 0.5 * rho * W2 * chord[i] * Ct       # [N/m]

    # Differential thrust and torque from each annulus:
    dT = B * Fn * dr                                  # thrust contribution [N]
    dQ = B * Ft * r * dr                              # torque contribution [N·m]

    # Total thrust, torque, power:
    T = float(np.sum(dT))                             # total thrust [N]
    Q = float(np.sum(dQ))                             # total torque [N·m]
    P = float(omega * Q)                              # total power [W]

    # Rotor swept area and non-dimensional coefficients:
    A = math.pi * R * R                               # swept area [m^2]
    CP = P / (0.5 * rho * A * V0**3)                  # power coefficient [-]
    CT = T / (0.5 * rho * A * V0**2)                  # thrust coefficient [-]

    # Spanwise distribution table (useful for plots/appendix)
    dist = pd.DataFrame({
        "r_m": r,                                     # element radius [m]
        "dr_m": dr,                                   # element width [m]
        "chord_m": chord,                             # chord [m]
        "twist_deg": beta,                            # twist [deg]
        "a": a,                                       # axial induction [-]
        "a_prime": ap,                                # tangential induction [-]
        "phi_deg": np.degrees(phi),                   # inflow angle [deg]
        "alpha_deg": alpha,                           # angle of attack [deg]
        "F": F,                                       # Prandtl loss factor [-]
        "CL": CL,                                     # lift coeff [-]
        "CD": CD,                                     # drag coeff [-]
        "Fn_N_per_m": Fn,                             # normal force per span [N/m]
        "Ft_N_per_m": Ft,                             # tangential force per span [N/m]
        "dT_N": dT,                                   # thrust per annulus [N]
        "dQ_Nm": dQ                                   # torque per annulus [N·m]
    })

    # Summary row (easy for your Task 1(e) table)
    summary = {
        "V0_mps": float(V0),                           # wind speed [m/s]
        "rpm": float(rpm),                             # rotor speed [rev/min]
        "P_kW": float(P / 1000),                       # power [kW]
        "Q_Nm": float(Q),                              # torque [N·m]
        "T_N": float(T),                               # thrust [N]
        "CP": float(CP),                               # CP [-]
        "CT": float(CT),                               # CT [-]
    }

    return summary, dist


# =============================================================================
# 6) Verification sweep across wind speeds (Task 1e)
# =============================================================================
def verify(geom, polar, winds, tsr, pitch, n, outdir, outNm):
    """
    geom   : geometry dict
    polar  : polar dict
    winds  : list of wind speeds [m/s]
    tsr    : constant TSR used for all wind speeds [-]
    pitch  : collective pitch angle [deg]
    n      : number of elements [-]
    outdir : output directory for CSV files

    Returns:
      df : summary table for all wind speeds
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []                 # list of per-wind summary dicts
    dists = {}                # map wind speed -> distribution dataframe

    for V in winds:
        s, dist = solve_bem_point(
            geom=geom,
            polar=polar,
            V0=float(V),        # wind speed [m/s]
            tsr=float(tsr),     # TSR [-]
            pitch_deg=float(pitch),
            n=int(n)
        )
        rows.append(s)
        dists[float(V)] = dist

    # Build summary DataFrame
    df = pd.DataFrame(rows).sort_values("V0_mps").reset_index(drop=True)

    # Simple sanity checks
    if (df["CP"] > 0.593).any():
        raise ValueError("CP exceeded Betz limit (0.593). Check inputs / polars / settings.")
    if (df["P_kW"] < -1e-6).any():
        raise ValueError("Negative power found (unexpected for normal operation).")

    # Save summary CSV
    df.to_csv(outdir / outNm, index=False)

    # Save one representative spanwise distribution (middle wind speed)
    Vmid = sorted(dists.keys())[len(dists) // 2]
    dists[Vmid].to_csv(outdir / f"bem_distributions_V{Vmid:.1f}.csv", index=False)

    return df


# =============================================================================
# 7) CLI entry point
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    # CLI arguments (each variable explained in the help string)
    ap.add_argument("--geom", type=str,
                    default="RISO-A1-A18 Profile for 500kW Reference Turbine Blade.csv",
                    help="Path to blade geometry CSV (must include r (m), chord, twist).")

    ap.add_argument("--polar", type=str,
                    default="RISO-A1-A18-Lift-Drag-Characteristics.csv",
                    help="Path to airfoil polar file (degree, CL, CD).")

    ap.add_argument(
        "--winds", type=float, nargs=3,
        default=[4, 20, 0.25],
        metavar=("V_MIN", "V_MAX", "V_STEP"),
        help="Wind sweep as: V_MIN V_MAX V_STEP (inclusive). Example: --winds 4 16 1"
    )

    ap.add_argument("--tsr", type=float, default=7.5,
                    help="Constant tip-speed ratio lambda = Omega*R/V [-].") #WE WILL BE CHANGING THIS ONCE DECIDED

    ap.add_argument("--pitch", type=float, default=0.0,
                    help="Collective pitch angle [deg].")

    ap.add_argument("--n", type=int, default=60,
                    help="Number of blade elements (annuli).")

    ap.add_argument("--outdir", type=str, default="bem_outputs", #Change this if you want a different folder
                    help="Folder to save CSV outputs.")
    ap.add_argument("--outNm", type=str, default="bem_summary.csv",
		    help="File to save CSV outputs to.")

    args = ap.parse_args()

    # --- Convert [min,max,step] -> explicit wind list ---
    Vmin, Vmax, Vstep = args.winds
    if Vstep <= 0:
        raise ValueError("--winds step must be > 0")
    if Vmax < Vmin:
        raise ValueError("--winds max must be >= min")

    winds = np.arange(Vmin, Vmax + 0.5 * Vstep, Vstep).tolist()  # inclusive-ish

    # Load input files
    geom = load_geometry(args.geom)
    polar = load_polar(args.polar)

    # Run verification sweep (Task 1e)
    df = verify(
        geom=geom,
        polar=polar,
        winds=winds,
        tsr=args.tsr,
        pitch=args.pitch,
        n=args.n,
        outdir=args.outdir,
        outNm=args.outNm
    )

    # Print summary table
    print("\nVerification summary (constant TSR):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Print where it saved
    print(f"\nSaved summary CSV to: {Path(args.outdir) / args.outNm}")


    # =============================================================================
    # 8) Quick plots 
    # =============================================================================
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.ravel()

    x = df["V0_mps"]

    # 1) Wind speed vs RPM
    axs[0].plot(x, df["rpm"], marker="o")
    axs[0].set_ylabel("Rotor speed [rpm]")
    axs[0].grid(True)
    axs[0].set_title("V0 vs RPM")

    # 2) Wind speed vs Power
    axs[1].plot(x, df["P_kW"], marker="o")
    axs[1].set_ylabel("Power [kW]")
    axs[1].grid(True)
    axs[1].set_title("V0 vs Power")

    # 3) Wind speed vs Torque (Q_Nm)
    axs[2].plot(x, df["Q_Nm"], marker="o")
    axs[2].set_xlabel("Wind speed V0 [m/s]")
    axs[2].set_ylabel("Torque [N·m]")
    axs[2].grid(True)
    axs[2].set_title("V0 vs Torque (Q)")

    # 4) Wind speed vs Thrust (optional but useful)
    axs[3].plot(x, df["T_N"], marker="o")
    axs[3].set_xlabel("Wind speed V0 [m/s]")
    axs[3].set_ylabel("Thrust [N]")
    axs[3].grid(True)
    axs[3].set_title("V0 vs Thrust")

    fig.suptitle("BEM summary vs wind speed (constant TSR)")
    fig.tight_layout()

    # Save plot image alongside your CSV outputs
    plot_path = Path(args.outdir) / "bem_plots_4panel.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Saved plots to: {plot_path}")

    plt.show()
    

if __name__ == "__main__":
    main()


