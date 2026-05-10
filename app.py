"""
SigNoise Web — Interface Streamlit
RF Fingerprinting AIS — PFE 2025-2026
EV2 Hamza Zouine & EV2 Saad EL MAACHI
"""

import streamlit as st
import numpy as np
import os, json, warnings, tempfile, time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import gdown
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SigNoise — RF Fingerprinting",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE (identique à SigNoise desktop)
# ─────────────────────────────────────────────────────────────────────────────
PAL = {
    "teal":    "#1D9E75",
    "coral":   "#D85A30",
    "purple":  "#534AB7",
    "amber":   "#BA7517",
    "blue":    "#185FA5",
    "dark":    "#2C2C2A",
    "mid":     "#5F5E5A",
    "light":   "#F7F7F5",
    "white":   "#FFFFFF",
    "border":  "#D3D1C7",
    "marine":  "#0A1637",
    "gold":    "#C4A050",
    "unknown": "#7B5EA7",
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL — reproduit l'apparence SigNoise
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: {PAL['light']};
    color: {PAL['dark']};
}}

/* Header gradient */
.sn-header {{
    background: linear-gradient(to right, {PAL['marine']}, #1C3464);
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.sn-header-title {{
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
}}
.sn-header-sub {{
    font-size: 11px;
    color: #9FA8C0;
    margin: 4px 0 0 0;
}}
.sn-badge {{
    background: rgba(29,158,117,0.18);
    color: {PAL['teal']};
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(29,158,117,0.3);
}}
.sn-badge-warn {{
    background: rgba(186,117,23,0.15);
    color: {PAL['amber']};
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(186,117,23,0.3);
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {PAL['white']};
    border-radius: 10px 10px 0 0;
    border: 1px solid {PAL['border']};
    border-bottom: none;
    padding: 0 8px;
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    height: 46px;
    padding: 0 20px;
    font-size: 13px;
    font-weight: 500;
    color: {PAL['mid']};
    border-bottom: 3px solid transparent;
    background: transparent;
    border-radius: 0;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {PAL['marine']};
    background: {PAL['light']};
}}
.stTabs [aria-selected="true"] {{
    color: {PAL['marine']} !important;
    border-bottom: 3px solid {PAL['gold']} !important;
    font-weight: 600 !important;
    background: {PAL['white']} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background: {PAL['white']};
    border: 1px solid {PAL['border']};
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 24px;
}}

/* Buttons */
.stButton > button {{
    background-color: {PAL['teal']};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 24px;
    transition: background 0.2s;
}}
.stButton > button:hover {{
    background-color: #0F6E56;
    border: none;
}}
.stButton > button:disabled {{
    background-color: {PAL['border']};
    color: {PAL['mid']};
}}

/* Cards métriques */
.sn-metric-card {{
    background: {PAL['white']};
    border: 1px solid {PAL['border']};
    border-top: 4px solid {PAL['teal']};
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}}
.sn-metric-value {{
    font-size: 28px;
    font-weight: 700;
    color: {PAL['dark']};
    margin: 0;
}}
.sn-metric-label {{
    font-size: 10px;
    font-weight: 600;
    color: {PAL['mid']};
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin: 4px 0 0 0;
}}

/* Tableau résultats */
.sn-result-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.sn-result-table th {{
    background: {PAL['marine']};
    color: #C0BEB4;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.sn-result-table td {{
    padding: 9px 14px;
    border-bottom: 1px solid {PAL['border']};
    color: {PAL['dark']};
}}
.sn-result-table tr:hover td {{
    background: #E1F5EE;
}}

/* Pill statuts */
.pill-connu {{
    background: rgba(29,158,117,0.12);
    color: {PAL['teal']};
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}}
.pill-inconnu {{
    background: rgba(123,94,167,0.12);
    color: {PAL['unknown']};
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}}

/* Selectbox, inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {{
    border: 1px solid {PAL['border']};
    border-radius: 6px;
    background: {PAL['white']};
    color: {PAL['dark']};
    font-size: 13px;
}}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {{
    border-color: {PAL['teal']};
    box-shadow: 0 0 0 2px rgba(29,158,117,0.15);
}}

/* Progress bar */
.stProgress > div > div > div > div {{
    background-color: {PAL['teal']};
    border-radius: 5px;
}}

/* Section titles */
.sn-section-title {{
    font-size: 13px;
    font-weight: 600;
    color: {PAL['mid']};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid {PAL['border']};
    padding-bottom: 8px;
    margin-bottom: 16px;
}}

/* Log console */
.sn-log {{
    background: {PAL['dark']};
    color: #A8C4A0;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    padding: 14px;
    border-radius: 8px;
    max-height: 220px;
    overflow-y: auto;
    line-height: 1.6;
}}

/* Hide streamlit branding */
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding-top: 1rem; padding-bottom: 1rem;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE — IDs fichiers
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_IDS = {
    "final_model_v2.pkl":        "17bv_klsdnVAnpARawJE8_3H1-TeC6PPW",
    "scaler_v2.pkl":             "19V79fTNiCwvjSO4StGQhSs1AjP8hKBL7",
    "rfecv_v2.pkl":              "1UqfkD2hmHONh3PnGQqOvFTXosBUbdJ-H",
    "gmm_db.pkl":                "1l130fak0rx6Y7t1SRs59QQLi1dSB3gkt",
    "inconnus_db.json":          "1h6fAm-pVBbKMy9cHq258w5o6evuV_PQq",
    "model_meta.json":           "1A8Oa0Zp2mD8G4MxfN-B6CTq1NMHLjLXg",
    "acquisition_01_float32.bin":"11ImUt-_DZMrxg0_ZmpbUIYIeOaZj362P",
    "acquisition_02_float32.bin":"1yh_123-jNVfFTG7ngFLwpsVbGpvqrv7i",
    "acquisition_03_float32.bin":"1AmmlcIPN_s79xWavqumaVUBKtN3AVmEk",
}

PRESETS = {
    "📡 Acquisition 01 — Navire (MMSI 247320162)":  "acquisition_01_float32.bin",
    "🏢 Acquisition 02 — Station base (MMSI BS:002270000)": "acquisition_02_float32.bin",
    "🔀 Acquisition 03 — Émission simultanée":       "acquisition_03_float32.bin",
}

MODELS_DIR     = Path("models")
RECORDINGS_DIR = Path("recordings")
MODELS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TÉLÉCHARGEMENT GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────────────
def _drive_download_silent(filename, dest_dir):
    """Télécharge depuis Google Drive SANS aucun appel Streamlit (safe dans cache_resource)."""
    dest = Path(dest_dir) / filename
    if dest.exists():
        return dest, None
    file_id = DRIVE_IDS.get(filename)
    if not file_id:
        return None, f"ID Drive inconnu pour {filename}"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(dest), quiet=True, fuzzy=True)
        if dest.exists():
            return dest, None
        return None, f"{filename} : téléchargement échoué (fichier absent après download)"
    except Exception as e:
        return None, f"{filename} : {e}"

def download_from_drive(filename, dest_dir):
    """Télécharge avec spinner Streamlit — à appeler HORS de cache_resource."""
    dest = Path(dest_dir) / filename
    if dest.exists():
        return dest
    with st.spinner(f"⬇️ Téléchargement {filename} depuis Google Drive..."):
        result, err = _drive_download_silent(filename, dest_dir)
    if err:
        st.error(f"❌ {err}")
    return result

@st.cache_resource(show_spinner=False)
def load_models():
    """Charge les modèles une seule fois en mémoire.
    Télécharge depuis Drive si nécessaire — SANS appels Streamlit UI."""
    errors = []
    model, scaler, rfecv, gmm_db, meta = None, None, None, None, {}

    # Téléchargement silencieux (pas de st.spinner ici — interdit dans cache_resource)
    for fname in ["final_model_v2.pkl", "scaler_v2.pkl", "rfecv_v2.pkl",
                  "gmm_db.pkl", "model_meta.json"]:
        dest = Path("models") / fname
        if not dest.exists():
            _, err = _drive_download_silent(fname, "models")
            if err:
                errors.append(f"Drive : {err}")

    try:
        model = joblib.load("models/final_model_v2.pkl")
    except Exception as e:
        errors.append(f"Modèle RF : {e}")
    try:
        scaler = joblib.load("models/scaler_v2.pkl")
    except Exception as e:
        errors.append(f"Scaler : {e}")
    try:
        rfecv = joblib.load("models/rfecv_v2.pkl")
    except Exception as e:
        errors.append(f"RFECV : {e}")
    try:
        gmm_db = joblib.load("models/gmm_db.pkl")
    except Exception as e:
        errors.append(f"GMM : {e}")
    try:
        with open("models/model_meta.json") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    return model, scaler, rfecv, gmm_db, meta, errors

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SIGNAL (extrait de SigNoise_v2_5.py)
# ─────────────────────────────────────────────────────────────────────────────
def _load_iq_full(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    n = len(raw) // 8
    return np.frombuffer(raw[:n*8], dtype=np.complex64).copy()

def _preprocess_iq(iq, fs, decim=1):
    iq = iq - np.mean(iq)
    sq = iq ** 2
    cfo_est = np.angle(np.mean(sq[1:] * np.conj(sq[:-1]))) / (2 * np.pi) * (fs / 2)
    t = np.arange(len(iq)) / fs
    iq = iq * np.exp(-1j * 2 * np.pi * cfo_est * t)
    if decim > 1:
        from scipy.signal import decimate
        iq_r = decimate(iq.real, decim, ftype="fir", zero_phase=True)
        iq_i = decimate(iq.imag, decim, ftype="fir", zero_phase=True)
        iq = (iq_r + 1j * iq_i).astype(np.complex64)
    pwr = np.mean(np.abs(iq) ** 2)
    if pwr > 0:
        iq = iq / np.sqrt(pwr)
    return iq

def _detect_bursts(iq, fs, smooth_ms=1.0, threshold_db=8.0,
                   min_burst_ms=10.0, merge_gap_ms=2.0):
    EPS_E = 1e-30
    energie = np.abs(iq) ** 2
    win = max(1, int(smooth_ms * 1e-3 * fs))
    kern = np.ones(win) / win
    e_smooth = np.convolve(energie, kern, mode="same")
    e_db = 10 * np.log10(e_smooth + EPS_E)
    plancher = np.percentile(e_db, 10)
    seuil = plancher + threshold_db
    masque = e_db > seuil
    segs, en_burst, debut = [], False, 0
    for i, v in enumerate(masque):
        if v and not en_burst:
            debut = i; en_burst = True
        elif not v and en_burst:
            segs.append((debut, i)); en_burst = False
    if en_burst:
        segs.append((debut, len(masque)))
    gap_min = int(merge_gap_ms * 1e-3 * fs)
    fused = []
    for seg in segs:
        if fused and (seg[0] - fused[-1][1]) < gap_min:
            fused[-1] = (fused[-1][0], seg[1])
        else:
            fused.append(list(seg))
    min_samp = int(min_burst_ms * 1e-3 * fs)
    return [(s, e) for s, e in fused if (e - s) >= min_samp]

def iq_to_phase(iq):
    phase = np.unwrap(np.angle(iq))
    t = np.arange(len(phase))
    return phase - np.polyval(np.polyfit(t, phase, 1), t)

def moving_average(x, m):
    cs = np.cumsum(np.concatenate(([0.0], x.astype(float))))
    return (cs[m:] - cs[:-m]) / m

def compute_avar(phase, taus, fs):
    avar, N = [], len(phase)
    for tau in taus:
        m = int(round(tau * fs))
        if m < 1 or 2*m >= N: avar.append(np.nan); continue
        pa = moving_average(phase, m); n = len(pa) - 2*m
        if n <= 0: avar.append(np.nan); continue
        d = pa[2*m:2*m+n] - 2*pa[m:m+n] + pa[:n]
        avar.append(float(np.mean(d**2) / (2*tau**2)))
    return np.array(avar)

def compute_hvar(phase, taus, fs):
    hvar, N = [], len(phase)
    for tau in taus:
        m = int(round(tau * fs))
        if m < 1 or 3*m >= N: hvar.append(np.nan); continue
        pa = moving_average(phase, m); n = len(pa) - 3*m
        if n <= 0: hvar.append(np.nan); continue
        d = pa[3*m:3*m+n] - 3*pa[2*m:2*m+n] + 3*pa[m:m+n] - pa[:n]
        hvar.append(float(np.mean(d**2) / (6*tau**2)))
    return np.array(hvar)

def apply_emd(phase, max_imf=8, subsample=5000, use_cmse=True):
    try:
        from PyEMD import EMD as PyEMD
        N = len(phase)
        if N > subsample:
            idx_sub = np.linspace(0, N-1, subsample).astype(int)
            phase_sub = phase[idx_sub]
        else:
            idx_sub = np.arange(N)
            phase_sub = phase
        emd = PyEMD(); emd.MAX_ITERATION = 200
        imfs = emd.emd(phase_sub.astype(float), max_imf=max_imf)
        if imfs.ndim == 1:
            return phase.copy()
        if use_cmse:
            n_imfs, NL = len(imfs), imfs.shape[1]
            mse = np.zeros(n_imfs)
            recon = np.zeros(NL)
            for k in range(n_imfs - 1, -1, -1):
                recon += imfs[k]
                mse[k] = np.mean((phase_sub - recon) ** 2)
            var_sig = np.var(phase_sub) + 1e-20
            cmse = mse / var_sig
            cmse_ext = np.append(cmse, 0.0)
            gains = np.abs(np.array([cmse_ext[k+1] - cmse_ext[k] for k in range(n_imfs)]))
            gains_norm = gains / (gains.sum() + 1e-20)
            seuil_g = gains_norm.mean() + 0.5 * gains_norm.std()
            idx_bruit = np.where(gains_norm < seuil_g)[0]
            if len(idx_bruit) == 0:
                idx_bruit = np.array([0])
            bruit_sub = np.sum(imfs[idx_bruit], axis=0)
        else:
            energies = np.array([np.sum(i**2) for i in imfs])
            bruit_sub = imfs[int(np.argmax(energies[:-1])) if len(energies) > 1 else 0]
        bruit_full = np.interp(np.arange(N), idx_sub, bruit_sub)
        return bruit_full
    except Exception:
        return phase.copy()

from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks

def features_variance(phase, taus, fs):
    def loglog_pentes(taus, vals):
        valid = ~np.isnan(vals) & (vals > 0) & (taus > 0)
        if valid.sum() < 4: return np.zeros(3)
        lt, lv = np.log10(taus[valid]), np.log10(vals[valid])
        segs = np.array_split(np.arange(len(lt)), 3)
        return np.array([np.polyfit(lt[s], lv[s], 1)[0] if len(s) >= 2 else 0. for s in segs])
    def amps_at(taus, vals, targets):
        valid = ~np.isnan(vals) & (vals > 0) & (taus > 0)
        if valid.sum() < 2: return np.zeros(len(targets))
        lt, lv = np.log10(taus[valid]), np.log10(vals[valid])
        return np.array([np.interp(np.log10(t), lt, lv) for t in targets])
    def infl(taus, vals):
        valid = ~np.isnan(vals) & (vals > 0) & (taus > 0)
        if valid.sum() < 6: return 0.
        lt, lv = np.log10(taus[valid]), np.log10(vals[valid])
        d2 = np.gradient(np.gradient(lv, lt), lt)
        peaks, _ = find_peaks(np.abs(d2))
        if len(peaks) == 0: return 0.
        return float(lt[peaks[np.argmax(np.abs(d2[peaks]))]])
    target = [taus[len(taus)//5], taus[len(taus)//2], taus[4*len(taus)//5]]
    feats = []
    for fn in [compute_avar, compute_hvar]:
        v = fn(phase, taus, fs)
        feats.extend(loglog_pentes(taus, v).tolist())
        feats.extend(amps_at(taus, v, target).tolist())
        feats.append(infl(taus, v))
    return np.array(feats)

def full_feature_vector(iq_seg, taus, fs, use_emd=True, use_cmse=True,
                         emd_subsample=5000, emd_max_imf=8):
    phase_brute = iq_to_phase(iq_seg)
    phase_purif = apply_emd(phase_brute, max_imf=emd_max_imf,
                             subsample=emd_subsample, use_cmse=use_cmse) if use_emd else phase_brute.copy()
    f_var = features_variance(phase_purif, taus, fs)
    # PSD
    f_p, Pxx = welch(iq_seg, fs=fs, nperseg=1024, return_onesided=False)
    Pdb = 10*np.log10(np.abs(Pxx)+1e-20)
    f_pos = f_p[f_p >= 0]; P_pos = Pdb[f_p >= 0]
    n3 = max(2, len(f_pos)//3)
    f_psd = np.array([
        np.polyfit(f_pos[1:n3], P_pos[1:n3], 1)[0] if n3 > 2 else 0.,
        np.polyfit(f_pos[2*n3:], P_pos[2*n3:], 1)[0] if len(f_pos[2*n3:]) > 2 else 0.,
        float(f_pos[np.argmax(P_pos)]) if len(P_pos) > 0 else 0.,
        float(np.percentile(P_pos[2*n3:], 50)) if len(P_pos[2*n3:]) > 0 else 0.,
        float(np.max(P_pos) - np.median(P_pos)) if len(P_pos) > 0 else 0.,
    ])
    # Freq inst
    freq = np.diff(phase_brute) * fs / (2 * np.pi) / fs
    p5, p25, p75, p95 = np.percentile(freq, [5, 25, 75, 95])
    f_fi = np.array([float(np.mean(freq)), float(np.std(freq)),
                     float(skew(freq)), float(kurtosis(freq)),
                     float(p5), float(p25), float(p75), float(p95)])
    # Power
    pwr = (iq_seg.real**2 + iq_seg.imag**2).astype(np.float64)
    p10, p50, p90 = np.percentile(pwr, [10, 50, 90])
    mn = float(np.mean(pwr)); st = float(np.std(pwr))
    f_pw = np.array([mn, st, float(p10), float(p50), float(p90),
                     float(np.max(pwr)/(mn+1e-20)),
                     float((p90-p10)/(p50+1e-20))])
    # Phase HO
    seg_n = len(phase_purif)//8
    lv = [np.var(phase_purif[i*seg_n:(i+1)*seg_n]) for i in range(8)] if seg_n > 0 else [0]*8
    p_nz = np.histogram(phase_purif, bins=50, density=True)[0] + 1e-20
    f_ph = np.array([float(np.mean(lv)), float(np.std(lv)),
                     float(skew(phase_purif)), float(kurtosis(phase_purif)),
                     float(-np.sum(p_nz * np.log2(p_nz)))])
    # EMD features
    try:
        from PyEMD import EMD as PyEMD
        ph_s = phase_brute[np.linspace(0, len(phase_brute)-1, min(emd_subsample, len(phase_brute))).astype(int)]
        emd = PyEMD(); emd.MAX_ITERATION = 200
        imfs = emd.emd(ph_s.astype(float), max_imf=emd_max_imf)
        if imfs.ndim == 1: imfs = imfs.reshape(1, -1)
        energies = np.array([np.sum(i**2) for i in imfs])
        total = energies.sum() + 1e-20
        idx_dom = int(np.argmax(energies[:-1])) if len(energies) > 1 else 0
        f_emd = np.array([float(energies[idx_dom]/total), float(energies[0]/total),
                          float(energies[-1]/total), float(idx_dom)])
    except Exception:
        f_emd = np.zeros(4)

    fv = np.concatenate([f_var, f_psd, f_fi, f_pw, f_ph, f_emd])
    return fv, phase_brute, phase_purif

def identifier_burst(fv, model, scaler, rfecv, gmm_db, seuil_proba=0.60):
    try:
        fv_s = scaler.transform(fv.reshape(1, -1))
        fv_r = rfecv.transform(fv_s)[0] if rfecv is not None else fv_s[0]
        probas = model.predict_proba(fv_r.reshape(1, -1))[0]
        idx_max = int(np.argmax(probas))
        classe_pred = model.classes_[idx_max]
        proba_max = float(probas[idx_max])
        log_lik = None
        est_connu = proba_max >= seuil_proba
        if est_connu and gmm_db is not None:
            gmm_info = gmm_db.get(classe_pred)
            if gmm_info is not None:
                log_lik = float(gmm_info["gmm"].score_samples(fv_r.reshape(1, -1))[0])
                if log_lik < gmm_info["seuil"]:
                    est_connu = False
        if not est_connu:
            classe_pred = "INCONNU"
        return classe_pred, proba_max, log_lik, fv_r, compute_avar(
            np.zeros(100), np.logspace(-4, -2, 30), 400e3)
    except Exception as e:
        return "ERREUR", 0.0, None, fv, np.zeros(30)

def run_pipeline(filepath, params, model, scaler, rfecv, gmm_db):
    """Lance le pipeline complet sur un fichier IQ et retourne les résultats."""
    fs = params["fs_original"]
    decim = params.get("decim", 2)
    fs_eff = fs // decim
    taus = np.logspace(
        np.log10(50e-6), np.log10(3e-3),
        params.get("n_taus", 30)
    )

    iq = _load_iq_full(filepath)
    iq = _preprocess_iq(iq, fs, decim=decim)
    bursts = _detect_bursts(
        iq, fs_eff,
        smooth_ms=params.get("burst_smooth_ms", 1.0),
        threshold_db=params.get("burst_threshold_db", 8.0),
        min_burst_ms=params.get("burst_min_ms", 10.0),
        merge_gap_ms=params.get("burst_merge_gap_ms", 2.0),
    )
    max_b = params.get("max_bursts", 50)
    bursts = sorted(bursts, key=lambda se: se[1]-se[0], reverse=True)[:max_b]
    bursts = sorted(bursts, key=lambda se: se[0])

    results = []
    avars_list = []
    phases_list = []

    for i, (s, e) in enumerate(bursts):
        seg = iq[s:e].copy()
        fv, phase_brute, phase_purif = full_feature_vector(
            seg, taus, fs_eff,
            use_emd=params.get("use_emd", True),
            use_cmse=params.get("use_cmse", True),
            emd_subsample=params.get("emd_sub", 5000),
            emd_max_imf=params.get("emd_imf", 8),
        )
        fv = np.nan_to_num(fv)
        classe, proba, log_lik, fv_r, _ = identifier_burst(
            fv, model, scaler, rfecv, gmm_db,
            seuil_proba=params.get("seuil_proba", 0.60)
        )
        avar = compute_avar(phase_purif, taus, fs_eff)
        hvar = compute_hvar(phase_purif, taus, fs_eff)
        avars_list.append((taus, avar, hvar, classe))
        phases_list.append(phase_brute)
        results.append({
            "burst": i + 1,
            "début_s": round(s / fs_eff, 4),
            "durée_ms": round((e - s) / fs_eff * 1000, 2),
            "classe": classe,
            "probabilité": round(proba, 4),
            "log_lik": round(log_lik, 3) if log_lik is not None else "—",
            "statut": "CONNU" if classe != "INCONNU" else "INCONNU",
        })
        yield i + 1, len(bursts), results[-1], avars_list, phases_list

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
model, scaler, rfecv, gmm_db, meta, load_errors = load_models()

model_ok = model is not None
badge_html = (
    f'<span class="sn-badge">✓ Modèle chargé — {len(model.classes_)} classes</span>'
    if model_ok else
    '<span class="sn-badge-warn">⚠ Modèle non chargé</span>'
)

st.markdown(f"""
<div class="sn-header">
  <div>
    <p class="sn-header-title">📡 SigNoise — RF Fingerprinting</p>
    <p class="sn-header-sub">Projet de Fin d'Études — EV2 Hamza Zouine &amp; EV2 Saad EL MAACHI — 2025–2026</p>
  </div>
  <div>{badge_html}</div>
</div>
""", unsafe_allow_html=True)

if load_errors:
    with st.expander("⚠️ Erreurs de chargement", expanded=False):
        for e in load_errors:
            st.error(e)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tab_id, tab_inc, tab_sig, tab_train, tab_params, tab_about = st.tabs([
    "📡  Identification",
    "👥  Inconnus",
    "📊  Analyse Signal",
    "🚂  Entraînement",
    "⚙️  Paramètres",
    "ℹ️  À propos",
])

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "results"      not in st.session_state: st.session_state.results = []
if "avars"        not in st.session_state: st.session_state.avars = []
if "phases"       not in st.session_state: st.session_state.phases = []
if "last_params"  not in st.session_state: st.session_state.last_params = {}

# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_id:

    # Bandeau modèle
    if model_ok:
        classes_str = " · ".join(model.classes_)
        n_feat = model.n_features_in_
        f1_str = f"F1 = {meta.get('f1_macro', '?')}" if meta else ""
        st.markdown(f"""
        <div style="background:{PAL['marine']};border-radius:8px;padding:12px 18px;
                    display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
          <div>
            <span style="color:#FFFFFF;font-size:13px;font-weight:600;">
              Modèle : Random Forest 300 estimateurs
            </span><br>
            <span style="color:#9FA8C0;font-size:11px;">
              Classes : {classes_str} &nbsp;|&nbsp; Features : {n_feat} &nbsp;|&nbsp; {f1_str}
            </span>
          </div>
          <span class="sn-badge">RFECV ✓ &nbsp; GMM ✓</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Modèle non chargé — vérifiez la connexion Google Drive dans les logs.")

    st.markdown('<p class="sn-section-title">Source du fichier IQ</p>', unsafe_allow_html=True)

    col_src, col_params = st.columns([1.6, 1])

    with col_src:
        source_mode = st.radio(
            "Mode source", ["📂 Enregistrement prédéfini", "⬆️ Importer mon fichier .bin"],
            horizontal=True, label_visibility="collapsed"
        )

        filepath_to_use = None

        if source_mode == "📂 Enregistrement prédéfini":
            preset_label = st.selectbox("Sélectionner l'enregistrement", list(PRESETS.keys()))
            preset_file = PRESETS[preset_label]
            dest = RECORDINGS_DIR / preset_file
            col_dl, col_info = st.columns([1, 2])
            with col_dl:
                if st.button("⬇️ Charger depuis Drive", use_container_width=True):
                    with st.spinner(f"Téléchargement {preset_file}..."):
                        result = download_from_drive(preset_file, RECORDINGS_DIR)
                    if result:
                        st.success(f"✓ {preset_file} prêt")
                    else:
                        st.error("Échec du téléchargement")
            with col_info:
                if dest.exists():
                    size_mb = dest.stat().st_size / 1e6
                    st.markdown(f"""
                    <div style="padding:8px 12px;background:{PAL['light']};border-radius:6px;
                                border-left:3px solid {PAL['teal']};margin-top:4px;">
                      <span style="font-size:12px;color:{PAL['dark']};">
                        ✓ <strong>{preset_file}</strong> — {size_mb:.1f} Mo
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
                    filepath_to_use = str(dest)
                else:
                    st.markdown(f"""
                    <div style="padding:8px 12px;background:#FFF8EE;border-radius:6px;
                                border-left:3px solid {PAL['amber']};margin-top:4px;">
                      <span style="font-size:12px;color:{PAL['amber']};">
                        ⬇ Fichier non présent — cliquez "Charger depuis Drive"
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            uploaded = st.file_uploader(
                "Glissez votre fichier IQ float32 (.bin)",
                type=["bin"],
                help="Format : IQ float32 entrelacé (complex64) — USRP / EM200"
            )
            if uploaded:
                tmp_path = Path(tempfile.mkdtemp()) / uploaded.name
                tmp_path.write_bytes(uploaded.read())
                filepath_to_use = str(tmp_path)
                size_mb = tmp_path.stat().st_size / 1e6
                st.success(f"✓ {uploaded.name} chargé — {size_mb:.1f} Mo")

    with col_params:
        st.markdown('<p class="sn-section-title">Paramètres</p>', unsafe_allow_html=True)
        fs_orig     = st.number_input("Fs original (Hz)", value=800000, step=100000, format="%d")
        decim       = st.selectbox("Décimation", [1, 2, 4, 8], index=1)
        max_bursts  = st.slider("Max bursts", 5, 100, 30)
        seuil_proba = st.slider("Seuil probabilité", 0.40, 0.95, 0.60, 0.05)
        use_emd     = st.checkbox("EMD + CMSE", value=True)

    params = {
        "fs_original": fs_orig,
        "decim": decim,
        "max_bursts": max_bursts,
        "seuil_proba": seuil_proba,
        "use_emd": use_emd,
        "use_cmse": True,
        "burst_smooth_ms": 1.0,
        "burst_threshold_db": 8.0,
        "burst_min_ms": 10.0,
        "burst_merge_gap_ms": 2.0,
        "n_taus": 30,
        "emd_sub": 5000,
        "emd_imf": 8,
    }

    st.markdown("---")
    col_btn, col_abort = st.columns([1, 4])
    with col_btn:
        run_btn = st.button(
            "▶  Lancer l'identification",
            disabled=(not model_ok or filepath_to_use is None),
            use_container_width=True,
        )

    if run_btn and model_ok and filepath_to_use:
        st.session_state.results = []
        st.session_state.avars   = []
        st.session_state.phases  = []
        st.session_state.last_params = params

        st.markdown('<p class="sn-section-title">Progression</p>', unsafe_allow_html=True)
        progress_bar  = st.progress(0)
        progress_text = st.empty()
        log_area      = st.empty()
        log_lines     = []

        try:
            for i, n_total, res, avars, phases in run_pipeline(
                filepath_to_use, params, model, scaler, rfecv, gmm_db
            ):
                st.session_state.results = [r for r in st.session_state.results] + [res]
                st.session_state.avars   = avars
                st.session_state.phases  = phases
                progress_bar.progress(int(i / n_total * 100))
                pill = "🟢" if res["statut"] == "CONNU" else "🟣"
                progress_text.markdown(
                    f"**Burst {i}/{n_total}** — {pill} `{res['classe']}` — proba {res['probabilité']:.3f}"
                )
                log_lines.append(
                    f"Burst {i:3d}/{n_total} → {res['classe']:<20}  proba={res['probabilité']:.3f}"
                    + (f"  log-lik={res['log_lik']}" if res['log_lik'] != '—' else "")
                )
                log_area.markdown(
                    f'<div class="sn-log">{"<br>".join(log_lines[-12:])}</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Erreur pipeline : {e}")

        progress_bar.progress(100)
        progress_text.markdown("✅ **Analyse terminée**")

    # ── Résultats ──────────────────────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results
        st.markdown('<p class="sn-section-title" style="margin-top:24px">Résultats</p>',
                    unsafe_allow_html=True)

        # Métriques
        n_total   = len(results)
        n_connus  = sum(1 for r in results if r["statut"] == "CONNU")
        n_inconnus = n_total - n_connus
        classes_det = list({r["classe"] for r in results if r["classe"] != "INCONNU"})
        proba_moy = round(np.mean([r["probabilité"] for r in results]), 3)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl, color in [
            (c1, n_total,   "BURSTS TRAITÉS",  PAL["teal"]),
            (c2, n_connus,  "CONNUS",           PAL["teal"]),
            (c3, n_inconnus,"INCONNUS",         PAL["unknown"]),
            (c4, f"{proba_moy:.3f}", "PROBA MOY.", PAL["gold"]),
        ]:
            col.markdown(f"""
            <div class="sn-metric-card" style="border-top-color:{color}">
              <p class="sn-metric-value" style="color:{color}">{val}</p>
              <p class="sn-metric-label">{lbl}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tableau
        rows_html = ""
        for r in results:
            pill = f'<span class="pill-connu">{r["statut"]}</span>' if r["statut"] == "CONNU" \
                   else f'<span class="pill-inconnu">{r["statut"]}</span>'
            rows_html += f"""
            <tr>
              <td><strong>{r['burst']}</strong></td>
              <td>{r['début_s']} s</td>
              <td>{r['durée_ms']} ms</td>
              <td><strong>{r['classe']}</strong></td>
              <td>{r['probabilité']}</td>
              <td>{r['log_lik']}</td>
              <td>{pill}</td>
            </tr>"""

        st.markdown(f"""
        <table class="sn-result-table">
          <thead><tr>
            <th>#</th><th>Début</th><th>Durée</th>
            <th>Classe</th><th>Probabilité</th><th>Log-lik</th><th>Statut</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # Export CSV
        st.markdown("<br>", unsafe_allow_html=True)
        df_export = pd.DataFrame(results)
        st.download_button(
            "⬇️ Exporter CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="signoise_resultats.csv",
            mime="text/csv",
        )

# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — INCONNUS
# ═════════════════════════════════════════════════════════════════════════════
with tab_inc:
    st.markdown('<p class="sn-section-title">Émetteurs inconnus détectés (open set)</p>',
                unsafe_allow_html=True)

    inconnus_path = Path("models/inconnus_db.json")
    if not inconnus_path.exists():
        _drive_download_silent("inconnus_db.json", "models")

    inconnus_db = {}
    if inconnus_path.exists():
        try:
            with open(inconnus_path) as f:
                inconnus_db = json.load(f)
        except Exception:
            pass

    # Inconnus de la session courante
    if st.session_state.results:
        session_inconnus = [r for r in st.session_state.results if r["classe"] == "INCONNU"]
        if session_inconnus:
            st.info(f"🔍 **{len(session_inconnus)} burst(s) inconnu(s)** détectés dans la dernière analyse.")

    if not inconnus_db:
        st.markdown(f"""
        <div style="text-align:center;padding:48px;color:{PAL['mid']}">
          <div style="font-size:40px;margin-bottom:12px">👥</div>
          <p style="font-size:15px;font-weight:600">Aucun inconnu enregistré</p>
          <p style="font-size:13px">Les émetteurs non reconnus apparaîtront ici après une analyse</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for uid, data in inconnus_db.items():
            if data.get("statut") == "INTÉGRÉ":
                continue
            with st.expander(f"🟣 {uid} — {data.get('n', 0)} captures"):
                col_a, col_b = st.columns(2)
                col_a.metric("Captures", data.get("n", 0))
                col_a.metric("Statut", data.get("statut", "?"))
                col_b.metric("Premier contact", data.get("date_premier", "?")[:10])
                col_b.metric("Dernier contact", data.get("date_dernier", "?")[:10])

# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — ANALYSE SIGNAL
# ═════════════════════════════════════════════════════════════════════════════
with tab_sig:
    st.markdown('<p class="sn-section-title">Analyse des signatures de bruit</p>',
                unsafe_allow_html=True)

    if not st.session_state.avars:
        st.markdown(f"""
        <div style="text-align:center;padding:48px;color:{PAL['mid']}">
          <div style="font-size:40px;margin-bottom:12px">📊</div>
          <p style="font-size:15px;font-weight:600">Aucune donnée à afficher</p>
          <p style="font-size:13px">Lancez une identification pour visualiser les signatures</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        avars = st.session_state.avars
        results = st.session_state.results

        # Graphique AVAR par classe
        fig_avar = go.Figure()
        colors_map = {}
        color_list = [PAL["teal"], PAL["coral"], PAL["purple"],
                      PAL["amber"], PAL["blue"], PAL["gold"]]
        for i, (taus, avar, hvar, classe) in enumerate(avars):
            if classe not in colors_map:
                colors_map[classe] = (PAL["unknown"] if classe == "INCONNU"
                                      else color_list[len(colors_map) % len(color_list)])
            col = colors_map[classe]
            valid = ~np.isnan(avar) & (avar > 0)
            if valid.sum() > 2:
                fig_avar.add_trace(go.Scatter(
                    x=taus[valid], y=avar[valid],
                    mode="lines",
                    name=f"Burst {i+1} — {classe}",
                    line=dict(color=col, width=1.2),
                    opacity=0.6,
                    showlegend=(i < 15),
                ))

        fig_avar.update_layout(
            title=dict(text="Variance d'Allan (AVAR) par burst", font=dict(size=14, color=PAL["dark"])),
            xaxis=dict(type="log", title="τ (s)", gridcolor=PAL["border"],
                       title_font=dict(size=12), tickfont=dict(size=11)),
            yaxis=dict(type="log", title="AVAR", gridcolor=PAL["border"],
                       title_font=dict(size=12), tickfont=dict(size=11)),
            paper_bgcolor=PAL["light"],
            plot_bgcolor=PAL["light"],
            legend=dict(font=dict(size=10), bgcolor=PAL["white"],
                        bordercolor=PAL["border"], borderwidth=1),
            height=380,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_avar, use_container_width=True)

        # Camembert des classes
        if results:
            from collections import Counter
            counts = Counter(r["classe"] for r in results)
            fig_pie = go.Figure(go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.45,
                marker=dict(colors=[
                    colors_map.get(c, PAL["mid"]) for c in counts.keys()
                ]),
                textfont=dict(size=12),
            ))
            fig_pie.update_layout(
                title=dict(text="Distribution des classes détectées",
                           font=dict(size=14, color=PAL["dark"])),
                paper_bgcolor=PAL["light"],
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(font=dict(size=11)),
            )
            col_pie, col_bar = st.columns(2)
            with col_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_bar:
                # Probabilités par burst
                df_r = pd.DataFrame(results)
                fig_bar = px.bar(
                    df_r, x="burst", y="probabilité", color="classe",
                    color_discrete_map=colors_map,
                    title="Probabilité par burst",
                    labels={"burst": "Burst #", "probabilité": "Probabilité"},
                )
                fig_bar.update_layout(
                    paper_bgcolor=PAL["light"],
                    plot_bgcolor=PAL["light"],
                    height=320,
                    margin=dict(l=40, r=20, t=40, b=40),
                    legend=dict(font=dict(size=10)),
                )
                fig_bar.add_hline(y=params.get("seuil_proba", 0.60) if st.session_state.last_params
                                    else 0.60,
                                   line_dash="dash", line_color=PAL["coral"],
                                   annotation_text="Seuil", annotation_font_size=10)
                st.plotly_chart(fig_bar, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — ENTRAÎNEMENT
# ═════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown('<p class="sn-section-title">Entraînement du modèle</p>',
                unsafe_allow_html=True)

    st.info("""
    ℹ️ **L'entraînement s'effectue en local** sur votre machine avec `SigNoise_v2_5.py` 
    (application desktop PyQt6), puis les fichiers `.pkl` résultants sont uploadés sur Google Drive 
    et rechargés automatiquement par cette interface.
    """)

    st.markdown(f"""
    <div style="background:{PAL['light']};border:1px solid {PAL['border']};border-radius:8px;padding:16px;margin-top:16px;">
      <p style="font-size:13px;font-weight:600;color:{PAL['dark']};margin:0 0 10px">Pipeline d'entraînement (SigNoise desktop)</p>
      <ol style="font-size:13px;color:{PAL['mid']};margin:0;padding-left:20px;line-height:2">
        <li>Chargement des acquisitions IQ (acq01 + acq02)</li>
        <li>Prétraitement : DC · CFO · décimation · normalisation</li>
        <li>Détection bursts AIS + extraction features (AVAR · HVAR · DSP · FI · EMD)</li>
        <li>Cross-validation 5-fold (SVM · RF · GBT · k-NN)</li>
        <li>RFECV — sélection automatique des features discriminantes</li>
        <li>Entraînement final Random Forest 300 estimateurs</li>
        <li>GMM par classe — calibration z-score Z=2 (open set)</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if model_ok:
        st.success(f"✅ Modèle actif : **Random Forest** — {model.n_features_in_} features — classes : `{'` · `'.join(model.classes_)}`")
    else:
        st.error("❌ Aucun modèle chargé")

# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 5 — PARAMÈTRES
# ═════════════════════════════════════════════════════════════════════════════
with tab_params:
    st.markdown('<p class="sn-section-title">Paramètres avancés</p>',
                unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Acquisition**")
        st.number_input("Fs original (Hz)", value=800000, key="p_fs", format="%d")
        st.selectbox("Décimation", [1, 2, 4, 8], index=1, key="p_decim")
        st.markdown("**Détection bursts**")
        st.number_input("Seuil énergie (dB)", value=8.0, key="p_thresh", step=0.5)
        st.number_input("Durée min burst (ms)", value=10.0, key="p_minburst", step=1.0)
        st.number_input("Merge gap (ms)", value=2.0, key="p_merge", step=0.5)
    with col_p2:
        st.markdown("**Pipeline signal**")
        st.checkbox("EMD (Empirical Mode Decomposition)", value=True, key="p_emd")
        st.checkbox("CMSE (sélection IMFs bruit)", value=True, key="p_cmse")
        st.number_input("Max IMF", value=8, key="p_maximf", step=1)
        st.number_input("EMD subsample", value=5000, key="p_sub", step=500)
        st.markdown("**Classification open set**")
        st.slider("Seuil probabilité", 0.40, 0.95, 0.60, 0.05, key="p_sproba")
        st.slider("GMM Z-score", 1.0, 4.0, 2.0, 0.5, key="p_zgmm")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{PAL['light']};border:1px solid {PAL['border']};border-radius:8px;padding:12px 16px;">
      <p style="font-size:12px;color:{PAL['mid']};margin:0;">
        💡 Les paramètres modifiés ici s'appliquent à la prochaine analyse. 
        Les valeurs par défaut correspondent à la configuration optimale validée sur acq01/acq02/acq03.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 6 — À PROPOS
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(f"""
    <div style="background:linear-gradient(to right,{PAL['marine']},#1C3464);
                border-radius:12px;padding:24px 28px;margin-bottom:20px;">
      <p style="color:#fff;font-size:18px;font-weight:700;margin:0">SigNoise — RF Fingerprinting</p>
      <p style="color:#9FA8C0;font-size:12px;margin:6px 0 0">
        Projet de Fin d'Études — EV2 Hamza Zouine &amp; EV2 Saad EL MAACHI — 2025–2026
      </p>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">

      <div style="background:{PAL['white']};border:1px solid {PAL['border']};
                  border-radius:8px;padding:16px;">
        <p style="font-size:13px;font-weight:600;color:{PAL['dark']};margin:0 0 10px">
          🎯 Objectif
        </p>
        <p style="font-size:13px;color:{PAL['mid']};line-height:1.7;margin:0">
          Identifier automatiquement des émetteurs AIS par leur 
          <strong>signature de bruit intrinsèque</strong> — les imperfections 
          uniques de l'oscillateur interne — sans décoder le contenu des messages.
        </p>
      </div>

      <div style="background:{PAL['white']};border:1px solid {PAL['border']};
                  border-radius:8px;padding:16px;">
        <p style="font-size:13px;font-weight:600;color:{PAL['dark']};margin:0 0 10px">
          🔬 Matériel
        </p>
        <p style="font-size:13px;color:{PAL['mid']};line-height:1.7;margin:0">
          Récepteurs SDR : <strong>USRP-2930 (EM200)</strong><br>
          Fréquence centrale : 162 MHz (AIS)<br>
          Fs : 800 kHz · Format : IQ float32 (complex64)
        </p>
      </div>

      <div style="background:{PAL['white']};border:1px solid {PAL['border']};
                  border-radius:8px;padding:16px;">
        <p style="font-size:13px;font-weight:600;color:{PAL['dark']};margin:0 0 10px">
          ⚙️ Pipeline
        </p>
        <p style="font-size:13px;color:{PAL['mid']};line-height:1.7;margin:0">
          Prétraitement → Détection bursts → Hilbert → EMD+CMSE → 
          AVAR/HVAR/MAVAR → Features DSP/FI/Puissance → RFECV → RF 300 est. → GMM open set
        </p>
      </div>

      <div style="background:{PAL['white']};border:1px solid {PAL['border']};
                  border-radius:8px;padding:16px;">
        <p style="font-size:13px;font-weight:600;color:{PAL['dark']};margin:0 0 10px">
          🤖 Classification
        </p>
        <p style="font-size:13px;color:{PAL['mid']};line-height:1.7;margin:0">
          Random Forest 300 estimateurs · RFECV · Cross-val 5-fold<br>
          Open set : proba ≥ seuil <strong>ET</strong> GMM log-lik ≥ seuil_z<br>
          Inconnus regroupés par distance relative 2σ
        </p>
      </div>

    </div>
    """, unsafe_allow_html=True)
