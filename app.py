import io
import os
import re
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Rectangle
from PIL import Image

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CHOICES = ["gemini-3.8-flash", "gemini-2.5-flash"]

FALLBACK_STOCKS = {
    "2330": {"code": "2330", "name": "台積電", "market": "TW"},
    "2454": {"code": "2454", "name": "聯發科", "market": "TW"},
    "2317": {"code": "2317", "name": "鴻海", "market": "TW"},
    "3711": {"code": "3711", "name": "日月光投控", "market": "TW"},
    "3037": {"code": "3037", "name": "欣興", "market": "TW"},
    "8046": {"code": "8046", "name": "南電", "market": "TW"},
    "2492": {"code": "2492", "name": "華新科", "market": "TW"},
    "2408": {"code": "2408", "name": "南亞科", "market": "TW"},
    "5274": {"code": "5274", "name": "信驊", "market": "TWO"},
    "6515": {"code": "6515", "name": "穎崴", "market": "TWO"},
    "3163": {"code": "3163", "name": "波若威", "market": "TWO"},
    "2880": {"code": "2880", "name": "華南金", "market": "TW"},
    "2881": {"code": "2881", "name": "富邦金", "market": "TW"},
    "2882": {"code": "2882", "name": "國泰金", "market": "TW"},
    "2884": {"code": "2884", "name": "玉山金", "market": "TW"},
    "2886": {"code": "2886", "name": "兆豐金", "market": "TW"},
    "2891": {"code": "2891", "name": "中信金", "market": "TW"},
    "2887": {"code": "2887", "name": "台新金", "market": "TW"},
    "0050": {"code": "0050", "name": "元大台灣50", "market": "TW"},
    "0056": {"code": "0056", "name": "元大高股息", "market": "TW"},
    "00878": {"code": "00878", "name": "國泰永續高股息", "market": "TW"},
    "00915": {"code": "00915", "name": "凱基優選高股息30", "market": "TW"},
}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tw_stock_directory() -> dict[str, dict]:
    return dict(FALLBACK_STOCKS)


def resolve_stock_query(query: str, stock_dir: dict[str, dict]) -> tuple[dict, list[dict]]:
    q = query.strip()
    if not q:
        raise ValueError("請輸入股票代號或股票名稱。")

    code_prefix_match = re.match(r"^([0-9A-Za-z]{4,6})(?:\.(?:TW|TWO))?(?:[\s\u3000]+.*)?$", q, re.IGNORECASE)
    if code_prefix_match:
        cand_code = code_prefix_match.group(1).upper()
        if cand_code in stock_dir:
            chosen = stock_dir[cand_code]
            return chosen, [chosen]
        fallback_info = {"code": cand_code, "name": cand_code, "market": ""}
        return fallback_info, [fallback_info]

    exact_matches = [info for info in stock_dir.values() if info["name"] == q]
    if exact_matches:
        return exact_matches[0], exact_matches

    q_lower = q.lower()
    fuzzy_matches = [
        info
        for info in stock_dir.values()
        if q_lower in info["name"].lower() or q_lower in info["code"].lower()
    ]
    if fuzzy_matches:
        fuzzy_matches.sort(key=lambda x: (0 if x["name"].startswith(q) else 1, len(x["code"])))
        return fuzzy_matches[0], fuzzy_matches[:8]

    raise ValueError(f"找不到「{q}」，請直接輸入 4～6 碼股票代號（例如 2330、00915）或常用名稱。")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_tw_stock_cached(code: str, name: str, market: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    code_up = code.strip().upper()
    candidates = [f"{code_up}.TWO", f"{code_up}.TW"] if market == "TWO" else [f"{code_up}.TW", f"{code_up}.TWO"]
    last_err = ""
    for symbol in candidates:
        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
                if len(df) >= 20:
                    return df, symbol
        except Exception as e:
            last_err = str(e)
    raise ValueError(f"找不到「{code_up} {name}」的行情資料（{last_err or '請確認代號或稍後再試'}）。")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    vol = d["Volume"].astype(float)

    d["MA5"] = close.rolling(5).mean()
    d["MA20"] = close.rolling(20).mean()
    d["MA60"] = close.rolling(60).mean()

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = ((close - low9) / (high9 - low9).replace(0, np.nan)) * 100
    rsv = rsv.fillna(50.0)

    k_vals, d_vals = [], []
    k_prev, d_prev = 50.0, 50.0
    for val in rsv:
        k_cur = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * val
        d_cur = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k_cur
        k_vals.append(k_cur)
        d_vals.append(d_cur)
        k_prev, d_prev = k_cur, d_cur

    d["K"] = k_vals
    d["D"] = d_vals

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["DIF"] = ema12 - ema26
    d["DEA"] = d["DIF"].ewm(span=9, adjust=False).mean()
    d["OSC"] = d["DIF"] - d["DEA"]
    d["VOL5"] = vol.rolling(5).mean()
    return d


def find_pivots(series: pd.Series, order: int = 3) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    vals = series.to_numpy(dtype=float)
    highs, lows = [], []
    n = len(vals)
    for i in range(order, n - order):
        window = vals[i - order : i + order + 1]
        if vals[i] == np.max(window) and np.sum(window == vals[i]) == 1:
            highs.append((i, float(vals[i])))
        if vals[i] == np.min(window) and np.sum(window == vals[i]) == 1:
            lows.append((i, float(vals[i])))
    return highs, lows


def detect_support_and_resistance_touches(
    df: pd.DataFrame,
    pivot_highs: list[tuple[int, float]],
    pivot_lows: list[tuple[int, float]],
    box_top: float,
    box_bot: float,
) -> dict:
    high = df["High"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(df)
    last_close = float(close[-1])

    recent_lows = [(idx, val) for idx, val in pivot_lows if idx >= max(0, n - 45)]
    double_support = None
    if len(recent_lows) >= 2:
        for i in range(len(recent_lows) - 1, 0, -1):
            idx2, val2 = recent_lows[i]
            for j in range(i - 1, -1, -1):
                idx1, val1 = recent_lows[j]
                if idx2 - idx1 < 4:
                    continue
                diff_pct = abs(val2 - val1) / max(min(val1, val2), 1e-6) * 100.0
                mid_high = float(np.max(high[idx1 : idx2 + 1]))
                rebound_pct = (mid_high - min(val1, val2)) / max(min(val1, val2), 1e-6) * 100.0
                if diff_pct <= 2.5 and rebound_pct >= 2.0:
                    avg_sup = (val1 + val2) / 2.0
                    double_support = {
                        "idx1": idx1,
                        "val1": val1,
                        "idx2": idx2,
                        "val2": val2,
                        "avg": avg_sup,
                        "label": f"雙支撐（第1低 {val1:.2f}、第2低 {val2:.2f}，均價約 {avg_sup:.2f}）",
                    }
                    break
            if double_support:
                break

    support_candidates = []
    if double_support:
        support_candidates.append(double_support["avg"])
    if recent_lows:
        support_candidates.append(recent_lows[-1][1])
    if box_bot > 0:
        support_candidates.append(box_bot)
    ma20_last = float(df["MA20"].iloc[-1]) if "MA20" in df and pd.notna(df["MA20"].iloc[-1]) else np.nan
    if pd.notna(ma20_last) and ma20_last <= last_close * 1.03:
        support_candidates.append(ma20_last)

    below_supports = [s for s in support_candidates if s <= last_close * 1.015]
    if below_supports:
        support_label = f"支撐參考約 {max(below_supports):.2f}"
    elif support_candidates:
        support_label = f"下方支撐參考約 {min(support_candidates):.2f}（現價略低於參考位）"
    else:
        support_label = "暫無明確支撐參考"

    win_start = max(0, n - 25)
    sub_high = high[win_start:]
    max_recent_high = float(np.max(sub_high))
    res_ref = max(max_recent_high, box_top) if box_top > 0 else max_recent_high
    touch_tol = res_ref * 0.018

    touch_indices = [win_start + k for k in range(len(sub_high)) if abs(sub_high[k] - res_ref) <= touch_tol]
    clustered_touches = []
    for idx in touch_indices:
        if not clustered_touches or idx - clustered_touches[-1] > 1:
            clustered_touches.append(idx)

    not_broken_cleanly = last_close <= res_ref * 1.015
    persistent_resistance = len(clustered_touches) >= 3 and not_broken_cleanly

    if persistent_resistance:
        res_label = f"一直碰到壓力（近25期於 {res_ref:.2f} 附近觸壓 {len(clustered_touches)} 次，尚未有效站穩突破）"
    elif len(clustered_touches) == 2 and not_broken_cleanly:
        res_label = f"雙壓／兩度觸壓（近25期於 {res_ref:.2f} 附近觸及 2 次，留意上方反壓）"
    else:
        res_label = f"上方壓力參考約 {res_ref:.2f}"

    return {
        "support_label": support_label,
        "double_support": double_support,
        "persistent_resistance": persistent_resistance,
        "res_label": res_label,
        "res_ref": res_ref,
        "touch_indices": clustered_touches,
    }


def analyze_structure_and_equal_move(df: pd.DataFrame, box_lookback: int = 20, swing_order: int = 3) -> dict:
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    n = len(df)

    last_close = float(close[-1])
    prev_close = float(close[-2]) if n >= 2 else last_close
    chg_pct = ((last_close - prev_close) / prev_close * 100.0) if prev_close else 0.0

    pivot_highs, _ = find_pivots(df["High"], order=swing_order)
    _, pivot_lows_from_low = find_pivots(df["Low"], order=swing_order)

    higher_highs = len(pivot_highs) >= 2 and pivot_highs[-1][1] > pivot_highs[-2][1]
    higher_lows = len(pivot_lows_from_low) >= 2 and pivot_lows_from_low[-1][1] > pivot_lows_from_low[-2][1]

    if higher_highs and higher_lows:
        structure_label = "峰峰高＋底底高（偏多波段結構）"
    elif higher_highs:
        structure_label = "峰峰高延續（波段高點墊高）"
    else:
        structure_label = "尚未形成明確峰峰高"

    lb = min(box_lookback, n - 2)
    box_slice_high = high[-(lb + 1) : -1]
    box_slice_low = low[-(lb + 1) : -1]
    box_top = float(np.max(box_slice_high))
    box_bot = float(np.min(box_slice_low))
    box_height = max(box_top - box_bot, 0.0)
    box_range_pct = (box_height / box_bot * 100.0) if box_bot > 0 else 0.0

    is_box = box_range_pct <= 18.0 and box_height > 0
    if last_close > box_top:
        box_status = f"突破箱頂（箱區 {box_bot:.2f}～{box_top:.2f}，箱高 {box_height:.2f}）"
    elif last_close < box_bot:
        box_status = f"跌破箱底（箱區 {box_bot:.2f}～{box_top:.2f}，箱高 {box_height:.2f}）"
    elif is_box:
        box_status = f"盤區箱型震盪中（{box_bot:.2f}～{box_top:.2f}，振幅 {box_range_pct:.1f}%）"
    else:
        box_status = f"非典型窄幅箱型（近{lb}期高低 {box_bot:.2f}～{box_top:.2f}）"

    sr_info = detect_support_and_resistance_touches(df, pivot_highs, pivot_lows_from_low, box_top, box_bot)

    win_start = max(0, n - 12)
    sub_low = low[win_start:]
    min_idx_rel = int(np.argmin(sub_low))
    min_idx = win_start + min_idx_rel
    min_val = float(sub_low[min_idx_rel])

    pre_high_val = float(np.max(high[max(0, min_idx - 8) : min_idx + 1])) if min_idx > 0 else float(high[0])
    drop_pct = ((pre_high_val - min_val) / pre_high_val * 100.0) if pre_high_val > 0 else 0.0
    rebound_from_low_pct = ((last_close - min_val) / min_val * 100.0) if min_val > 0 else 0.0
    recovery_ratio = ((last_close - min_val) / (pre_high_val - min_val)) if (pre_high_val - min_val) > 0 else 0.0

    bars_after_low = (n - 1) - min_idx
    if drop_pct >= 5.0 and 1 <= bars_after_low <= 6 and recovery_ratio >= 0.75:
        pattern_label = f"V轉型態（自低點 {min_val:.2f} 快速收復 {recovery_ratio*100:.0f}% 跌幅）"
    elif drop_pct >= 4.0 and bars_after_low >= 1 and rebound_from_low_pct >= 2.5:
        pattern_label = f"反彈走勢（自低點 {min_val:.2f} 反彈 +{rebound_from_low_pct:.1f}%，收復跌幅 {recovery_ratio*100:.0f}%）"
    else:
        pattern_label = "一般波段波動（未見明顯急跌V轉或反彈）"

    equal_move_notes = []
    if box_height > 0:
        if last_close > box_top:
            equal_move_notes.append(f"箱型向上等幅：突破箱頂 {box_top:.2f}，1:1 滿足目標約 {box_top + box_height:.2f}")
        elif last_close < box_bot:
            equal_move_notes.append(f"箱型向下等幅：跌破箱底 {box_bot:.2f}，1:1 滿足目標約 {max(box_bot - box_height, 0.0):.2f}")
        else:
            equal_move_notes.append(
                f"箱型等幅預估：向上突破 {box_top:.2f} 約 {box_top + box_height:.2f}；向下跌破 {box_bot:.2f} 約 {max(box_bot - box_height, 0.0):.2f}"
            )

    if len(pivot_highs) >= 2 and len(pivot_lows_from_low) >= 2:
        prev_h = pivot_highs[-2][1]
        prev_l = pivot_lows_from_low[-2][1]
        cur_l = pivot_lows_from_low[-1][1]
        wave_move = max(prev_h - prev_l, 0.0)
        if wave_move > 0 and cur_l > prev_l:
            equal_move_notes.append(f"波段對稱等幅：前段漲幅 {wave_move:.2f}，自低點 {cur_l:.2f} 推算約 {cur_l + wave_move:.2f}")

    return {
        "last_close": last_close,
        "chg_pct": chg_pct,
        "structure_label": structure_label,
        "box_top": box_top,
        "box_bot": box_bot,
        "box_height": box_height,
        "box_status": box_status,
        "pattern_label": pattern_label,
        "equal_move_notes": equal_move_notes,
        "sr_info": sr_info,
    }


def render_chart_figure(df: pd.DataFrame, struct: dict, display_name: str, symbol: str, box_lb: int, show_n: int):
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]}
    )
    n_show = min(show_n, len(df))
    offset = len(df) - n_show
    sub = df.iloc[-n_show:].copy()
    x = np.arange(len(sub))

    opens = sub["Open"].to_numpy(dtype=float)
    highs = sub["High"].to_numpy(dtype=float)
    lows = sub["Low"].to_numpy(dtype=float)
    closes = sub["Close"].to_numpy(dtype=float)
    vols = sub["Volume"].to_numpy(dtype=float)

    for i in range(len(sub)):
        is_up = closes[i] >= opens[i]
        color = "#D92B2B" if is_up else "#1E8E3E"
        ax_price.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.0)
        body_low = min(opens[i], closes[i])
        body_h = max(abs(closes[i] - opens[i]), 0.05)
        ax_price.add_patch(Rectangle((i - 0.32, body_low), 0.64, body_h, facecolor=color, edgecolor=color, linewidth=0.8))

    ax_price.plot(x, sub["MA5"].to_numpy(dtype=float), label="MA5", linewidth=1.1, color="#F59E0B")
    ax_price.plot(x, sub["MA20"].to_numpy(dtype=float), label="MA20", linewidth=1.3, color="#2563EB")
    if sub["MA60"].notna().any():
        ax_price.plot(x, sub["MA60"].to_numpy(dtype=float), label="MA60", linewidth=1.3, color="#7C3AED")

    box_top, box_bot, box_h = struct["box_top"], struct["box_bot"], struct["box_height"]
    if box_h > 0:
        lb = min(box_lb, len(sub) - 1)
        start_x = max(0, len(sub) - 1 - lb)
        ax_price.hlines([box_top, box_bot], xmin=start_x, xmax=len(sub) - 1, colors="#0F766E", linestyles="--", linewidth=1.2)
        ax_price.axhline(box_top + box_h, color="#0F766E", linestyle=":", linewidth=0.9, alpha=0.7)

    sr = struct["sr_info"]
    ds = sr["double_support"]
    if ds and ds["idx1"] >= offset and ds["idx2"] >= offset:
        p1_x, p2_x = ds["idx1"] - offset, ds["idx2"] - offset
        ax_price.scatter([p1_x, p2_x], [ds["val1"], ds["val2"]], color="#0284C7", s=36, zorder=5)
        ax_price.plot([p1_x, p2_x], [ds["val1"], ds["val2"]], color="#0284C7", linestyle="-.", linewidth=1.3)

    if sr["persistent_resistance"] and sr["touch_indices"]:
        res_y = sr["res_ref"]
        ax_price.axhline(res_y, color="#B91C1C", linestyle="-.", linewidth=1.1, alpha=0.85)
        touch_xs = [idx - offset for idx in sr["touch_indices"] if idx >= offset]
        touch_ys = [float(df["High"].iloc[idx]) for idx in sr["touch_indices"] if idx >= offset]
        if touch_xs:
            ax_price.scatter(touch_xs, touch_ys, color="#B91C1C", marker="^", s=36, zorder=5)

    code_only = display_name.split()[0] if display_name else symbol
    ax_price.set_title(f"{code_only} ({symbol}) K-Line & Structure", fontsize=11)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, linestyle=":", alpha=0.4)

    vol_colors = ["#D92B2B" if closes[i] >= opens[i] else "#1E8E3E" for i in range(len(sub))]
    ax_vol.bar(x, vols, color=vol_colors, width=0.6, alpha=0.8)
    if sub["VOL5"].notna().any():
        ax_vol.plot(x, sub["VOL5"].to_numpy(dtype=float), color="#2563EB", linewidth=1.1, label="VOL5")
    ax_vol.set_ylabel("Volume", fontsize=9)
    ax_vol.grid(True, linestyle=":", alpha=0.4)

    step = max(1, len(sub) // 8)
    tick_pos = list(range(0, len(sub), step))
    ax_vol.set_xticks(tick_pos)
    ax_vol.set_xticklabels([sub.index[i].strftime("%m/%d") for i in tick_pos], fontsize=8)

    fig.tight_layout(pad=1.2)
    return fig


def prepare_image_bytes(pil_image: Image.Image, max_side: int = 1400) -> bytes:
    img = pil_image.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def analyze_image_with_gemini(api_key: str, pil_image: Image.Image, extra_prompt: str, preferred_model: str) -> str:
    if not api_key.strip():
        raise ValueError("請先輸入 Gemini API Key。")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip(), http_options=types.HttpOptions(timeout=45000))
    img_bytes = prepare_image_bytes(pil_image)

    system_instruction = (
        "你是一位擅長台股技術分析的助手。請根據使用者提供的K線截圖，以繁體中文分點清楚列出："
        "1. 盤面趨勢與均線/指標狀況；2. 支撐、雙支撐、一直碰到壓力標示；"
        "3. 反彈、V轉、峰峰高、盤區箱型等結構型態；4. 漲跌等幅度觀察；5. 策略觀察重點與風險提醒。"
    )
    user_text = "請分析這張K線截圖的指標、支撐/雙支撐/一直碰到壓力、結構型態與漲跌等幅度策略。"
    if extra_prompt.strip():
        user_text += f"\n補充條件：{extra_prompt.strip()}"

    models_to_try = [m for m in [preferred_model] + MODEL_CHOICES if m]
    models_to_try = list(dict.fromkeys(models_to_try))

    last_err = None
    for model_name in models_to_try:
        for _ in range(2):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), user_text],
                    config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2),
                )
                if resp.text:
                    return resp.text
            except Exception as e:
                last_err = e
                err_str = str(e)
                if any(k in err_str for k in ("503", "UNAVAILABLE", "429", "504", "DEADLINE_EXCEEDED", "timed out", "Timeout")):
                    time.sleep(1.5)
                    continue
                if "404" in err_str or "NOT_FOUND" in err_str:
                    break
                raise
    raise RuntimeError(f"模型呼叫失敗：{last_err}")


def main():
    st.set_page_config(page_title="K線技術分析與結構等幅工具（試用版）", layout="wide")
    st.title("K線技術分析與結構等幅工具（不公開試用版）")

    trial_pw = st.secrets.get("TRIAL_PASSWORD", "") if hasattr(st, "secrets") else ""
    if trial_pw:
        if "authed" not in st.session_state:
            st.session_state.authed = False
        if not st.session_state.authed:
            pw_input = st.text_input("請輸入試用通關密碼", type="password")
            if st.button("進入試用"):
                if pw_input == trial_pw:
                    st.session_state.authed = True
                    st.rerun()
                else:
                    st.error("密碼錯誤。")
            st.stop()

    with st.sidebar:
        st.header("分析設定")
        default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
        api_key = st.text_input("Gemini API Key（截圖分析用）", value=default_key, type="password")
        model_name = st.selectbox("分析模型", MODEL_CHOICES, index=0)
        box_lookback = st.number_input("箱型觀察期數", min_value=5, max_value=80, value=20, step=1)
        show_n = st.slider("圖表顯示 K 棒數", min_value=20, max_value=150, value=90, step=5)
        st.caption("※ 僅供研究參考，不構成投資建議。")

    tab1, tab2 = st.tabs(["代號／名稱分析", "K線截圖分析"])

    with tab1:
        col_q, col_p, col_i = st.columns([3, 1, 1])
        query = col_q.text_input("股票代號或名稱", value="2330 台積電")
        period = col_p.selectbox("資料區間", ["3mo", "6mo", "1y", "2y"], index=1)
        interval = col_i.selectbox("K線週期", ["1d", "1wk"], index=0)

        if st.button("計算指標與結構分析", type="primary"):
            try:
                stock_dir = fetch_tw_stock_directory()
                stock_info, _ = resolve_stock_query(query, stock_dir)
                display_name = f"{stock_info['code']} {stock_info['name']}".strip()
                with st.spinner(f"正在查詢 {display_name} 行情..."):
                    df, symbol = fetch_tw_stock_cached(
                        stock_info["code"], stock_info["name"], stock_info.get("market", ""), period, interval
                    )
                df_ind = compute_indicators(df)
                struct = analyze_structure_and_equal_move(df_ind, box_lookback=int(box_lookback), swing_order=3)

                fig = render_chart_figure(df_ind, struct, display_name, symbol, int(box_lookback), int(show_n))
                st.pyplot(fig, use_container_width=True)

                sr = struct["sr_info"]
                st.subheader(f"分析摘要：{display_name}（最新收盤 {struct['last_close']:.2f}｜單日 {struct['chg_pct']:+.2f}%）")
                st.markdown(
                    f"- **波段結構**：{struct['structure_label']}\n"
                    f"- **型態判讀**：{struct['pattern_label']}\n"
                    f"- **盤區箱型**：{struct['box_status']}\n"
                    f"- **支撐標示**：{sr['support_label']}\n"
                    + (f"- **雙支撐標示**：{sr['double_support']['label']}\n" if sr["double_support"] else "")
                    + f"- **壓力標示**：{sr['res_label']}"
                )
                if struct["equal_move_notes"]:
                    st.markdown("**漲跌等幅度提示**")
                    for n in struct["equal_move_notes"]:
                        st.markdown(f"- {n}")
            except Exception as e:
                st.error(str(e))

    with tab2:
        uploaded = st.file_uploader("上傳 K 線截圖（PNG / JPG / WEBP）", type=["png", "jpg", "jpeg", "webp"])
        extra = st.text_input("補充策略／關注點（選填）", placeholder="例如：看是否有雙支撐或一直碰到壓力")
        if uploaded is not None:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="已載入截圖", use_container_width=True)
            if st.button("AI 分析截圖指標與型態", type="primary"):
                try:
                    with st.spinner("Gemini 分析中..."):
                        result = analyze_image_with_gemini(api_key, pil_img, extra, model_name)
                    st.markdown("### 分析結果")
                    st.markdown(result)
                except Exception as e:
                    st.error(str(e))


if __name__ == "__main__":
    main()
