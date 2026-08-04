# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  AIROS V4.1.3 — Inference API                                           ║
# ║  Model  : airos_v4_1_scripted.pt  (TorchScript)                        ║
# ║  Schema : v4.1  |  Assets: 20  |  Horizons: T+1/2/3/4                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, gc, json, time, logging
from pathlib import Path

import numpy  as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("airos")

# ── File paths (same directory as app.py) ─────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODEL_PATH    = BASE_DIR / "airos_v4_1_scripted.pt"
MANIFEST_PATH = BASE_DIR / "manifest_4.1.3.json"

# ── Globals populated at startup ──────────────────────────────────────────────
_model         = None   # TorchScript module
_engine        = None   # AIROSEngine instance
_asset_registry = {}    # asset → {"norm_mean": np.float32 (22,), "norm_std": np.float32 (22,)}
_config        = {}


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — FROZEN CONTRACT (identical to training Cells 2 & 3)
# ══════════════════════════════════════════════════════════════════════════════

def _ema_np(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(np.float32)


def _rolling_std_f32(x: np.ndarray, w: int) -> np.ndarray:
    xd  = x.astype(np.float64)
    mu  = uniform_filter1d(xd,    size=w, mode="nearest")
    mu2 = uniform_filter1d(xd**2, size=w, mode="nearest")
    return np.sqrt(np.maximum(mu2 - mu**2, 0.0)).astype(np.float32)


def build_candle_features(ohlc_df: pd.DataFrame) -> np.ndarray:
    """
    Returns (N, 14) float32.  Column contract FROZEN:
      0  body_ratio    1  upper_wick   2  lower_wick   3  close_pos
      4  direction     5  norm_range   6  atr_ratio
      7  mom5          8  mom14        9  vol_std5
     10  compression  11  trend_state
     12  swing_h_dist 13  swing_l_dist
    """
    o = ohlc_df["open"].to_numpy(np.float32)
    h = ohlc_df["high"].to_numpy(np.float32)
    l = ohlc_df["low"].to_numpy(np.float32)
    c = ohlc_df["close"].to_numpy(np.float32)
    N   = len(c)
    eps = np.float32(1e-7)

    prev_c = np.empty_like(c); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr     = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14  = _ema_np(tr, 14) + eps

    rng         = np.maximum(h - l, eps)
    body_ratio  = np.abs(c - o) / rng
    upper_wick  = (h - np.maximum(o, c)) / rng
    lower_wick  = (np.minimum(o, c) - l) / rng
    close_pos   = (c - l) / rng
    direction   = np.sign(c - o).astype(np.float32)
    norm_range  = rng / atr14
    atr_ratio   = atr14 / (np.abs(c) + eps)

    ret1 = np.empty(N, np.float32); ret1[0] = 0.0
    ret1[1:] = (c[1:] - c[:-1]) / (np.abs(c[:-1]) + eps)

    mom5  = np.zeros(N, np.float32)
    mom14 = np.zeros(N, np.float32)
    mom5[5:]   = (c[5:]  - c[:-5])  / (np.abs(c[:-5])  + eps) / (atr_ratio[5:]  + eps)
    mom14[14:] = (c[14:] - c[:-14]) / (np.abs(c[:-14]) + eps) / (atr_ratio[14:] + eps)

    vol_std5    = _rolling_std_f32(ret1, 5) / (atr_ratio + eps)
    avg_rng20   = uniform_filter1d(rng.astype(np.float64), size=20, mode="nearest").astype(np.float32)
    compression = rng / (avg_rng20 + eps)
    ema8        = _ema_np(c, 8)
    ema21       = _ema_np(c, 21)
    trend_state = (ema8 - ema21) / (atr14 + eps)
    swing_h_dist = (maximum_filter1d(h, size=20, mode="nearest") - c) / (atr14 + eps)
    swing_l_dist = (c - minimum_filter1d(l, size=20, mode="nearest")) / (atr14 + eps)

    feats = np.empty((N, 14), dtype=np.float32)
    feats[:, 0]  = body_ratio
    feats[:, 1]  = upper_wick
    feats[:, 2]  = lower_wick
    feats[:, 3]  = close_pos
    feats[:, 4]  = direction
    feats[:, 5]  = norm_range
    feats[:, 6]  = atr_ratio
    feats[:, 7]  = mom5
    feats[:, 8]  = mom14
    feats[:, 9]  = vol_std5
    feats[:, 10] = compression
    feats[:, 11] = trend_state
    feats[:, 12] = swing_h_dist
    feats[:, 13] = swing_l_dist
    np.clip(feats, -5.0, 5.0, out=feats)
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def build_regime_token(candle_feats_window: np.ndarray, token_dim: int = 22) -> np.ndarray:
    """
    Single-sequence regime token for inference.
    Exact replica of Cell 2 build_regime_token().
    Returns (token_dim,) float32 — values at [0:7], zeros at [7:token_dim].
    """
    recent  = candle_feats_window[-20:]
    ts      = float(np.mean(recent[:, 11]))   # trend_state
    cs      = float(np.mean(recent[:, 10]))   # compression
    vs      = float(np.mean(recent[:, 6]))    # atr_ratio
    sh      = float(recent[-1, 12])           # swing_h_dist
    sl      = float(recent[-1, 13])           # swing_l_dist

    trend   = float(np.tanh(ts))
    comp    = float(1.0 - np.clip(cs, 0.0, 2.0) / 2.0)
    hi_vol  = float(np.clip(vs * 10.0, 0.0, 1.0))
    n_res   = float(1.0 / (1.0 + max(sh, 0.0)))
    n_sup   = float(1.0 / (1.0 + max(sl, 0.0)))
    t_bias  = float(trend * (1.0 - comp))
    brk_rsk = float(comp * hi_vol)

    token = np.zeros(token_dim, dtype=np.float32)
    token[:7] = [trend, comp, hi_vol, n_res, n_sup, t_bias, brk_rsk]
    return token


def build_tick_features(
    tick_store_df:             pd.DataFrame,
    ohlc_df:                   pd.DataFrame,
    bucket_minutes:            float,
    tick_max_per_bucket:       int,
    tick_speed_norm_rate:      float = 10.0,
    tick_density_norm_minutes: float = 14.0,
) -> np.ndarray:
    """
    Returns (N_bars, 8) float32.  Column contract FROZEN:
      0  buyer_pressure  1  seller_pressure  2  delta
      3  tick_speed      4  tick_accel       5  tick_imbalance
      6  micro_vol       7  tick_density
    """
    eps    = np.float32(1e-7)
    N_bars = len(ohlc_df)

    ts_ns  = tick_store_df["timestamp"].to_numpy(np.int64)
    prices = tick_store_df["price"].to_numpy(np.float32)

    diff    = np.empty_like(prices); diff[0] = 0.0; diff[1:] = prices[1:] - prices[:-1]
    is_buy  = (diff > 0).astype(np.float32)
    is_sell = (diff < 0).astype(np.float32)

    prices64 = prices.astype(np.float64)
    ret      = np.zeros(len(prices), np.float64)
    nz       = prices64[:-1] != 0
    ret[1:]  = np.where(nz, diff[1:].astype(np.float64) / prices64[:-1], 0.0)
    ret      = ret.astype(np.float32)

    bar_ns  = ohlc_df.index.as_unit("ns").asi8
    bar_idx = np.searchsorted(bar_ns, ts_ns, side="right") - 1
    valid   = (bar_idx >= 0) & (bar_idx < N_bars)
    bar_idx = bar_idx[valid].astype(np.int32)
    is_buy  = is_buy[valid];  is_sell = is_sell[valid]
    ret_v   = ret[valid]

    count    = np.bincount(bar_idx, minlength=N_bars).astype(np.float32)
    buy_sum  = np.bincount(bar_idx, weights=is_buy,  minlength=N_bars).astype(np.float32)
    sell_sum = np.bincount(bar_idx, weights=is_sell, minlength=N_bars).astype(np.float32)
    ret_sum  = np.bincount(bar_idx, weights=ret_v,   minlength=N_bars).astype(np.float32)
    ret2_sum = np.bincount(bar_idx, weights=ret_v.astype(np.float64)**2,
                           minlength=N_bars).astype(np.float32)

    total  = count + eps
    bp     = buy_sum  / total
    sp     = sell_sum / total
    delta  = bp - sp
    imb    = np.clip((buy_sum - sell_sum) / total, -1.0, 1.0)
    tspeed = np.clip(count / (bucket_minutes * tick_speed_norm_rate), 0.0, 1.0)
    taccel = np.empty(N_bars, np.float32); taccel[0] = 0.0
    taccel[1:] = tspeed[1:] - tspeed[:-1]
    e_r  = ret_sum  / total
    e_r2 = ret2_sum / total
    mvol = np.clip(np.sqrt(np.maximum(e_r2 - e_r**2, 0.0)) * 1000.0, 0.0, 1.0)
    eff_max = float(tick_max_per_bucket) * (bucket_minutes / tick_density_norm_minutes)
    dens    = np.clip(count / max(eff_max, 1.0), 0.0, 1.0)

    out = np.empty((N_bars, 8), dtype=np.float32)
    out[:, 0] = bp;     out[:, 1] = sp;    out[:, 2] = delta
    out[:, 3] = tspeed; out[:, 4] = taccel; out[:, 5] = imb
    out[:, 6] = mvol;   out[:, 7] = dens
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET PAYLOAD PARSERS  (unchanged from V3.1)
# ══════════════════════════════════════════════════════════════════════════════

def parse_websocket_payload(payload: dict) -> dict:
    """Platform sends: {asset, period, history: [[ts, price, flag],...], candles: [[...], ...]}"""
    ticks = [{"timestamp": float(t[0]), "price": float(t[1]), "flag": int(t[2])}
             for t in payload.get("history", [])]
    candles = [{"open_time": int(c[0]), "open": float(c[1]), "close": float(c[2]),
                "high": float(c[3]), "low": float(c[4]),
                "tick_count": int(c[5]), "close_time": float(c[6])}
               for c in payload.get("candles", [])]
    return {
        "asset":   payload.get("asset", "UNKNOWN"),
        "period":  int(payload.get("period", 60)),
        "ticks":   ticks,
        "candles": candles,
    }


def build_ohlc_from_platform_candles(parsed_candles: list, period_seconds: float,
                                      threshold: float):
    if not parsed_candles:
        return None, 0, {}
    candles_sorted = sorted(parsed_candles, key=lambda c: c["open_time"])
    last    = candles_sorted[-1]
    elapsed = last["close_time"] - last["open_time"]
    thresh  = period_seconds * threshold
    info    = {}
    if elapsed < thresh:
        info = {"reason": f"elapsed {elapsed:.1f}s < threshold {thresh:.1f}s"}
        candles_sorted = candles_sorted[:-1]
        if not candles_sorted:
            return None, 1, info
        rows = [{"timestamp": pd.Timestamp(c["open_time"], unit="s", tz="UTC"),
                 "open": c["open"], "high": c["high"], "low": c["low"],
                 "close": c["close"], "volume": c["tick_count"]}
                for c in candles_sorted]
        return pd.DataFrame(rows).set_index("timestamp"), 1, info
    rows = [{"timestamp": pd.Timestamp(c["open_time"], unit="s", tz="UTC"),
             "open": c["open"], "high": c["high"], "low": c["low"],
             "close": c["close"], "volume": c["tick_count"]}
            for c in candles_sorted]
    return pd.DataFrame(rows).set_index("timestamp"), 0, info


def build_tick_df_from_parsed(parsed_ticks: list) -> pd.DataFrame:
    if not parsed_ticks:
        return pd.DataFrame(columns=["timestamp", "price", "flag"])
    df = pd.DataFrame(parsed_ticks)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE  (V4.1.3)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_adaptive_norm(market_bars_40: np.ndarray):
    mean = market_bars_40.mean(axis=0).astype(np.float32)
    std  = np.maximum(market_bars_40.std(axis=0).astype(np.float32), np.float32(1e-6))
    return mean, std


class AIROSEngine:
    """
    V4.1.3 Production Inference Engine.

    Key difference vs V3 engine:
      - Loads TorchScript model → output is a plain tuple (logits, confidence)
        NOT a NamedTuple. Access: out[0]=logits, out[1]=confidence.
      - Norm stats loaded from manifest JSON into asset_registry (no .npy files).
      - Adaptive normalization for any asset not in registry.
    """

    def __init__(self, jit_model, cfg: dict, device: torch.device, asset_registry: dict):
        self.model          = jit_model
        self.cfg            = cfg
        self.device         = device
        self.asset_registry = asset_registry        # pre-loaded from manifest

        self.seq            = cfg["sequence_length"]       # 40
        self.mod_seq        = cfg["model_seq_len"]          # 41
        self.token_dim      = cfg["token_dim"]              # 22
        self.thresh         = cfg["confidence_threshold"]   # 0.65
        self.ph             = cfg["primary_horizon"]        # 0  (T+1)
        self.hnames         = [f"T+{h}" for h in cfg["horizons"]]  # ["T+1","T+2","T+3","T+4"]
        self.bk_min         = cfg["tick_bucket_min_minutes"]  # 7
        self.bk_max         = cfg["tick_bucket_max_minutes"]  # 14
        self.bk_max_ticks   = cfg["tick_max_per_bucket"]       # 210
        self.period         = cfg["ohlc_resample_min"] * 60   # 60 s
        self.partial_thresh = cfg["partial_candle_threshold"]  # 0.90
        self.speed_norm     = cfg["tick_speed_norm_rate"]      # 10.0
        self.density_norm   = cfg["tick_density_norm_minutes"] # 14.0

        n_h   = len(cfg["horizons"])
        temps = cfg.get("calibration_temperature", [1.0] * n_h)
        self.temperature = torch.tensor(temps, dtype=torch.float32).view(1, n_h, 1).to(device)

    def _clamp_bk(self, bk: float) -> float:
        return float(max(self.bk_min, min(self.bk_max, bk)))

    def _get_norm_params(self, asset: str, market_bars_40: np.ndarray = None):
        """Priority: registry → adaptive (from 40-bar window) → noop."""
        clean = asset.split("_")[0].upper()
        if clean in self.asset_registry:
            rec = self.asset_registry[clean]
            return rec["norm_mean"], rec["norm_std"], "registry"
        if market_bars_40 is not None:
            m, s = _compute_adaptive_norm(market_bars_40)
            return m, s, "adaptive"
        return (np.zeros(self.token_dim, np.float32),
                np.ones(self.token_dim,  np.float32), "noop")

    @torch.no_grad()
    def predict_from_payload(self, payload: dict) -> dict:
        parsed = parse_websocket_payload(payload)
        asset  = parsed["asset"].split("_")[0].upper()

        ohlc_df, n_dropped, partial_info = build_ohlc_from_platform_candles(
            parsed["candles"], self.period, self.partial_thresh
        )
        if n_dropped:
            log.warning(f"[{asset}] Partial candle excluded: {partial_info.get('reason','')}")

        tick_df = build_tick_df_from_parsed(parsed["ticks"])

        if len(parsed["ticks"]) >= 2:
            t_span = parsed["ticks"][-1]["timestamp"] - parsed["ticks"][0]["timestamp"]
            bk     = self._clamp_bk(t_span / 60.0)
        else:
            bk = float(self.bk_min)

        result = self._run_inference(ohlc_df, tick_df, bk, asset)
        result["partial_candle_dropped"] = n_dropped > 0
        if n_dropped:
            result["partial_candle_info"] = partial_info
        return result

    @torch.no_grad()
    def _run_inference(self, ohlc_df, tick_df, bucket_minutes: float, asset: str) -> dict:
        if ohlc_df is None or len(ohlc_df) == 0:
            return {"error": "No OHLC data", "direction": "NO_TRADE", "confidence": 0.0}


        min_needed = self.seq                    # exactly 40 bars required
        if len(ohlc_df) < min_needed:
            return {"direction": "NO_TRADE",
                    "error": f"Need {min_needed} bars, got {len(ohlc_df)}",
                    "confidence": 0.0}

        # ── Candle features ──────────────────────────────────────────────────
        df_full = ohlc_df.iloc[-min_needed:].copy()
        cf_full = build_candle_features(df_full)
        cf      = cf_full[-self.seq:]       # (40, 14)
        df      = df_full.iloc[-self.seq:]

        # ── Tick features ─────────────────────────────────────────────────────
        if tick_df is not None and len(tick_df) >= 10:
            tick_trimmed = tick_df[tick_df["timestamp"] >= df.index[0]].copy()
            if len(tick_trimmed) >= 5:
                tf = build_tick_features(
                    tick_store_df             = tick_trimmed,
                    ohlc_df                   = df,
                    bucket_minutes            = self._clamp_bk(bucket_minutes),
                    tick_max_per_bucket       = self.bk_max_ticks,
                    tick_speed_norm_rate      = self.speed_norm,
                    tick_density_norm_minutes = self.density_norm,
                )
            else:
                tf = np.zeros((self.seq, 8), np.float32)
        else:
            tf = np.zeros((self.seq, 8), np.float32)

        # ── Assemble market bars (40, 22) and normalise ───────────────────────
        market_bars = np.concatenate([cf, tf], axis=1).astype(np.float32)  # (40, 22)
        np.nan_to_num(market_bars, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        norm_mean, norm_std, norm_src = self._get_norm_params(asset, market_bars)
        market_bars = np.clip((market_bars - norm_mean) / norm_std, -5.0, 5.0)

        # ── Build full input token (41, 22) ───────────────────────────────────
        # Position 0  : regime token (7 values, rest zeros) — NOT normalised
        # Positions 1-40: normalised market bars
        regime_tok = build_regime_token(cf, self.token_dim)
        tok_full   = np.zeros((self.mod_seq, self.token_dim), np.float32)
        tok_full[0, :]  = regime_tok
        tok_full[1:, :] = market_bars

        inp = torch.from_numpy(tok_full).unsqueeze(0).to(self.device)   # (1, 41, 22)

        # ── Forward pass ─────────────────────────────────────────────────────
        # TorchScript wrapper returns plain tuple: (logits, confidence)
        # out[0] = logits     shape (1, 4, 2)
        # out[1] = confidence shape (1, 4)   [raw, pre-calibration]
        t0  = time.perf_counter()
        out = self.model(inp)
        lat = (time.perf_counter() - t0) * 1000

        # Temperature calibration + softmax
        cal_logits = out[0] / self.temperature       # (1, 4, 2)
        probs      = F.softmax(cal_logits[0], -1).cpu().numpy()   # (4, 2)
        conf       = probs.max(-1)                   # (4,)

        ph_conf  = float(conf[self.ph])
        ph_dir   = "BUY" if probs[self.ph, 1] >= 0.5 else "SELL"
        decision = ph_dir if ph_conf >= self.thresh else "NO_TRADE"

        return {
            "asset":       asset,
            "direction":   decision,
            "confidence":  round(ph_conf, 4),
            "horizon":     self.hnames[self.ph],
            "norm_source": norm_src,
            "regime": {
                "trend":         round(float(regime_tok[0]), 4),
                "compressed":    round(float(regime_tok[1]), 4),
                "high_vol":      round(float(regime_tok[2]), 4),
                "near_res":      round(float(regime_tok[3]), 4),
                "near_sup":      round(float(regime_tok[4]), 4),
                "trend_bias":    round(float(regime_tok[5]), 4),
                "breakout_risk": round(float(regime_tok[6]), 4),
            },
            "horizons": {
                hn: {
                    "direction":  "BUY" if probs[h, 1] >= 0.5 else "SELL",
                    "buy_prob":   round(float(probs[h, 1]), 4),
                    "confidence": round(float(conf[h]), 4),
                }
                for h, hn in enumerate(self.hnames)
            },
            "latency_ms": round(lat, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AIROS V4.1.3", version="4.1.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global _model, _engine, _asset_registry, _config

    # ── Load manifest (contains CONFIG + all asset norms) ────────────────────
    log.info(f"Loading manifest: {MANIFEST_PATH}")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    _config = manifest["config"]
    _config["model_seq_len"] = _config["sequence_length"] + 1  # 41

    # Build asset registry from manifest — no .npy files needed
    for asset, rec in manifest.get("per_asset_norm", {}).items():
        _asset_registry[asset] = {
            "norm_mean": np.array(rec["norm_mean"], dtype=np.float32),
            "norm_std":  np.array(rec["norm_std"],  dtype=np.float32),
        }
    log.info(f"  Loaded norms for {len(_asset_registry)} assets: {sorted(_asset_registry)}")

    # ── Load TorchScript model ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading TorchScript model on {device}: {MODEL_PATH}")
    _model = torch.jit.load(str(MODEL_PATH), map_location=device)
    _model.eval()
    log.info("  Model loaded OK")

    # ── Warm-up pass (eliminates first-call latency) ─────────────────────────
    with torch.no_grad():
        _dummy = torch.zeros(1, _config["model_seq_len"], _config["token_dim"]).to(device)
        _ = _model(_dummy)
    log.info("  Warm-up pass OK")

    _engine = AIROSEngine(_model, _config, device, _asset_registry)
    log.info("AIROS V4.1.3 engine ready")
    gc.collect()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": "4.1.3",
        "schema":  "v4.1",
        "assets":  sorted(_asset_registry.keys()),
        "device":  "cuda" if torch.cuda.is_available() else "cpu",
        "model":   MODEL_PATH.name,
    }


@app.post("/predict")
def predict(payload: dict):
    """
    Accepts the platform WebSocket payload forwarded as HTTP POST JSON:
    {
      "asset":   "EURUSD",
      "period":  60,
      "history": [[timestamp, price, flag], ...],
      "candles": [[open_time, open, close, high, low, tick_count, close_time], ...]
    }
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    try:
        return _engine.predict_from_payload(payload)
    except Exception as exc:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
