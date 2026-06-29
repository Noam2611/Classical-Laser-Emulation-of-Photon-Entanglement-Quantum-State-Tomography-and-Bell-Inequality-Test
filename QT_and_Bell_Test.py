"""
QT_and_Bell_Test.py
============
Linear, function-based pipeline for processing Quantum-State-Tomography
(Part A) and CHSH Bell-inequality (Part B) lab videos.

Engine: per-frame mean red-channel intensity inside an ROI; frames whose
mean crosses a configurable threshold are stored as "active event frames".
Coincidences are found by a local-window scan (no global optimisation).
"""

import os
import re
import json
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.widgets import RectangleSelector


# ===========================================================
#  CONFIGURATION
# ===========================================================

# --- run control ---
DO_PART_A = False
DO_PART_B = False
SHOW_PLOTS = True

# --- Part A simulation (simulated signals, no real video needed) ---
RUN_SIMULATION_A = True   # True = run simulated signal simulation for Part A
SIM_A_SHOTS      = 500     # number of pulses to simulate
SIM_A_SEED       = 1      # random seed for reproducibility

# --- ROI cache control ---
RESELECT_ROI_A = False # True = overrule the saved json file if required
RESELECT_ROI_B = False # True = overrule the saved json file if required
ROI_FRAME_A = None   # None = auto-detect the first frame with red light
ROI_FRAME_B = None

# --- event detection (local intersection paradigm) ---
#   For each frame we compute the average red-channel intensity inside the ROI,
#   normalise to [0,1] over the whole clip, and call any frame above
#   EVENT_THRESHOLD an "active event frame".
EVENT_THRESHOLD = 0.75         # PART A: 0..1, fraction of normalised intensity
# EVENT_THRESHOLD = 0.89         # PART B: 0..1, fraction of normalised intensity
MIN_EVENT_SEPARATION = 30      # minimum frame gap between two distinct events

COINCIDENCE_WINDOW = 30        # PART A: ± frames searched in channel B around each A event
# COINCIDENCE_WINDOW = 12        # PART B: ± frames searched in channel B around each A event
# --- experiment-onset trimming ---
ONSET_RED_MEAN = 2000         # raw red-channel mean that marks "laser on"
ONSET_MAX_SEARCH = 3000        # only search this many leading frames

# --- "first red frame" detection (used to pick a frame for ROI selection) ---
# scan the leading frames of the clip and return the first one whose RAW whole-frame
# red sum exceeds RED_START_THRESHOLD. ROI selection then opens that exact frame.

RED_START_THRESHOLD = 2000     # raw red-pixel sum (whole frame) marking "experiment started"
MAX_START_SEARCH_FRAMES = 3000 # only look for the start within the first N frames

# --- Part A ---
PART_A_DIR = r"C:\Users\Noam\Desktop\Physics\Lab C\Tomography\parta"
PART_A_FILENAME_RE = r"(\d+)bits\.mp4" # for: 10,25,50
PART_A_ROI_NAMES = ("Alice V", "Alice H", "Bob V", "Bob H")
BASIS_LABELS = ("HH", "HV", "VH", "VV")
PART_A_ROI_FILE = "roi_cache_partA.json"

# --- Part B ---
PART_B_DIR = r"C:\Users\Noam\Desktop\Physics\Lab C\Tomography\partb"
# Accepts either "bit1_0_22.5.mp4" or "0to22.5.mp4" style filenames.
PART_B_FILENAME_RE = r"^(?:.*?_)?(-?\d+(?:\.\d+)?)(?:to|_)(-?\d+(?:\.\d+)?)\.mp4$"
PART_B_ROI_NAMES = ("Alice", "Bob")
PART_B_ROI_FILE = "roi_cache_partB_10.json"
ALPHA_ANGLES = (-45.0, 0.0, 45.0, 90.0)
BETA_ANGLES = (-22.5, 22.5, 67.5, 112.5)

# --- Part B per-video ROI overrides ---
# Most (alpha, beta) videos share the single global ROI cached in
# PART_B_ROI_FILE. If a specific clip needs its own ROI (e.g. Bob's spot
# drifted at one polariser angle), add a (alpha, beta) -> override-cache-file
# entry here. First time that pair is processed the user is prompted to draw
# the ROI interactively (on its own first-red frame); afterwards it's loaded
# from the override cache just like the global one.
PART_B_ROI_OVERRIDES: dict = {
    # Example:
    (-45.0, 67.5): "roi_cache_partB_-45_67.5.json",
    ( 45.0,  -22.5): "roi_cache_partB_45_-22.5_.json",
    ( 45.0,  67.5): "roi_cache_partB_45_67.5_.json",
    ( 45.0,  22.5): "roi_cache_partB_45_22.5_.json",
}

# CHSH evaluation angles (the four E(a,b) terms)
CHSH_PAIRS = ((0.0, 22.5), (0.0, 67.5), (45.0, 22.5), (45.0, 67.5))
CHSH_SIGNS = (+1, -1, +1, +1)

# ===========================================================
#  SMALL HELPERS

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalise(arr):
    """
    Min-max normalization - squeezes any array of numbers into the range [0, 1]
    by subtracting the minimum and dividing by the range.
    """
    arr = np.asarray(arr, dtype=np.float64)
    span = arr.max() - arr.min()
    if span <= 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span


def round_angle(a):
    """
    This exists purely to prevent floating-point precision issues
    when using angles as dictionary keys.
    """
    return round(float(a), 6)

#  VIDEO I/O

def open_capture(path):
    """
    calling cv2.VideoCapture amd to raise an error if
    the file can't be opened (wrong path, corrupt file, etc).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap

def grab_frame(path, idx):
    """
    Used when only one representative frame is needed (like in showing the user a frame for ROI selection).
    The cap.release() call properly frees the video file from memory, preventing resource leaks
    if the function is called many times across different videos.
    """
    cap = open_capture(path)
    if idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def roi_red_sum(frame_bgr, roi):
    """
    Takes one video frame and one ROI rectangle, crops the frame to that rectangle,
    select only red-channel and sum intensity inside the ROI (R is index 2 in BGR).
    """
    x, y, w, h = roi
    patch = frame_bgr[y:y + h, x:x + w, 2]
    return float(patch.sum()) if patch.size else 0.0

def measure_red_traces(path, rois):
    """
    Walk the video once and produce one red-mean trace per ROI.
    Returns (traces, frame_count) per ROI where traces is shape (n_rois, n_frames).
    A "trace" is a signal recorded over time
    """
    traces = [[] for _ in rois]
    cap = open_capture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for k, r in enumerate(rois):
                traces[k].append(roi_red_sum(frame, r))
    finally:
        cap.release()
    arrs = [np.asarray(t, dtype=np.float64) for t in traces]
    return arrs, (len(arrs[0]) if arrs else 0)


def _find_onset_index(sig, threshold, max_frames):
    """Internal helper: index of the first element in 'sig' >= 'threshold',
    scanning at most 'max_frames'; 0 if none found."""
    limit = min(len(sig), max_frames)
    for i in range(limit):
        if sig[i] >= threshold:
            return i
    return 0


#  EVENT VECTORISATION + LOCAL-WINDOW COINCIDENCES

def trace_to_event_frames(trace, threshold=EVENT_THRESHOLD,
                          min_gap=MIN_EVENT_SEPARATION):
    """
    Convert a 1-D intensity trace into a list of representative "event" frame
    indices. Two stages for a more robust method for pulse shapes that spread a few frames
    or don't have a sharp peak. The min-separation stage handles the rare case of two nearby pulses cleanly.

      Stage 1 — burst pass: normalise to [0,1]; finds all frames above the threshold. Then groups consecutive
      above-threshold frames that are within min_gap of each other into "bursts."
      Each burst represents one physical pulse event.
      From each burst, the single brightest frame is picked as the candidate event.

      Stage 2 — minimum-separation enforcement: for cases of separate pulses
                that are closer than min_gap. walk the candidates ordered
                by intensity (brightest first) and accept one only if it is
                at least 'min_gap' frames away from every already-accepted event.
                This guarantees the output list has no two events closer than min_gap frames apart.
    """
    if len(trace) == 0:
        return []
    norm = normalise(trace)
    above = np.where(norm >= threshold)[0]
    if above.size == 0:
        return []

    # --- Stage 1: burst grouping ---
    candidates = []
    burst_start = above[0]
    prev = above[0]
    for idx in above[1:]:
        if idx - prev <= min_gap:
            prev = idx
            continue
        seg = norm[burst_start:prev + 1]
        candidates.append(int(burst_start + int(np.argmax(seg))))
        burst_start = idx
        prev = idx
    seg = norm[burst_start:prev + 1]
    candidates.append(int(burst_start + int(np.argmax(seg))))

    # --- Stage 2: enforce min_gap between accepted events ---
    candidates.sort(key=lambda i: -norm[i])   # brightest first
    accepted = []
    for c in candidates:
        if all(abs(c - a) >= min_gap for a in accepted):
            accepted.append(c)
    accepted.sort()
    return accepted


def count_coincidences(events_a, events_b, window=COINCIDENCE_WINDOW):
    """
    LOCAL-WINDOW coincidence engine.

    For each event frame in channel A, scan channel B inside the window
    [a - window, a + window]. The first available (unused) B event in that
    neighbourhood is claimed as the partner. We advance past it so no B event
    is double-counted.
    """
    if not events_a or not events_b:
        return 0
    b_sorted = sorted(events_b)
    used = [False] * len(b_sorted)
    j_start = 0
    n_coinc = 0

    for a in sorted(events_a):
        # advance the left edge of the search window
        while j_start < len(b_sorted) and b_sorted[j_start] < a - window:
            j_start += 1
        j = j_start
        while j < len(b_sorted) and b_sorted[j] <= a + window:
            if not used[j]:
                used[j] = True
                n_coinc += 1
                break
            j += 1
    return n_coinc


#  ROI SELECTION (matplotlib RectangleSelector widget)

def find_first_red_frame(path, search_limit=MAX_START_SEARCH_FRAMES):
    """
    Public entry point for onset detection. Opens the video, computes the
    per-frame whole-frame HSV red-mask pixel sum, and delegates the actual
    "first frame above threshold" search to `_find_onset_index`. Returns
    the chosen frame index (the frame we want to show in the ROI window).

    The leading search window is normalised to [0, 1] so the cutoff scales
    with whatever baseline this particular clip has — robust against clips
    whose first frame already carries ambient red pixels.
    """
    cap = open_capture(path)
    reds = []
    try:
        for _ in range(search_limit):
            ok, frame = cap.read()
            if not ok:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
            m2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
            reds.append(int(np.sum(m1 | m2)))
    finally:
        cap.release()
    if not reds:
        return 0

    sig = np.asarray(reds, dtype=np.int64)
    raw_max = int(sig.max())
    raw_min = int(sig.min())

    if raw_max - raw_min < 1:
        print(f"  [first-red] '{os.path.basename(path)}': flat red signal "
              f"(max={raw_max}), defaulting to frame 0.")
        return 0

    norm = (sig.astype(np.float64) - raw_min) / (raw_max - raw_min)
    idx = _find_onset_index(norm, EVENT_THRESHOLD, search_limit)
    print(f"  [first-red] '{os.path.basename(path)}': chosen frame={idx} "
          f"(norm>= {EVENT_THRESHOLD}, raw max={raw_max}, raw min={raw_min})")
    return idx

def draw_one_roi(image_bgr, prompt):
    """Show one frame and let the user drag a rectangle. Returns (x,y,w,h)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_title(f"{prompt}\n(drag a box, ENTER=confirm, ESC=cancel)",
                 fontsize=10)

    pick = {"rect": None, "cancel": False}

    def on_select(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        pick["rect"] = (min(x0, x1), min(y0, y1),
                        abs(x1 - x0), abs(y1 - y0))

    selector = RectangleSelector(
        ax, on_select, useblit=False, interactive=True,
        button=[1], minspanx=2, minspany=2,
    )

    def on_key(ev):
        if ev.key == "enter":
            plt.close(fig)
        elif ev.key == "escape":
            pick["cancel"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    # keep reference so the linter doesn't complain
    _ = selector

    if pick["cancel"] or pick["rect"] is None:
        raise RuntimeError("ROI selection cancelled by user.")
    x, y, w, h = pick["rect"]
    return (int(round(x)), int(round(y)),
            max(1, int(round(w))), max(1, int(round(h))))


def gather_rois(path, names, frame_index):
    frame = grab_frame(path, frame_index)
    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} of {path}")
    print(f"  [ROI] Drawing {len(names)} ROIs on frame "
          f"{frame_index} of {os.path.basename(path)}")
    rois = []
    for name in names:
        rois.append(draw_one_roi(frame, f"ROI: {name}  (frame {frame_index})"))
    return rois

def load_rois_from_disk(cache_file, names):
    """
    Read ROIs from a JSON cache. Accepts both the new "names" key and the
    legacy "labels" key.
    """
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        stored_names = payload.get("names", payload.get("labels", []))
        if list(stored_names) != list(names):
            return None
        rois = [tuple(map(int, r)) for r in payload["rois"]]
        if len(rois) != len(names):
            return None
        return rois
    except Exception:
        return None

def save_rois_to_disk(cache_file, video_path, names, rois):
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump({
            "saved_at": timestamp(),
            "video_path": video_path,
            "names": list(names),
            "rois": [list(map(int, r)) for r in rois],
        }, fh, indent=2)


def acquire_rois(cache_file, video_path, names, force, frame_index):
    """
    Return ROI coords. Order of operations:
      1. If a valid cache exists and `force` is False, reuse it (no window).
      2. Otherwise resolve `frame_index`:
           - explicit int -> use that frame
           - None         -> call find_first_red_frame() to land on the first
                             frame where the laser is clearly on.
      3. Open that frame, let the user draw rectangles, save to cache.
    """
    if not force:
        cached = load_rois_from_disk(cache_file, names)
        if cached is not None:
            print(f"  [ROI] Re-using cached ROIs from '{cache_file}' "
                  f"(set the matching RESELECT flag True to redraw).")
            return cached
        print(f"  [ROI] No cache at '{cache_file}' — opening selection window.")
    else:
        print(f"  [ROI] Forced reselect for '{cache_file}'.")

    if frame_index is None:
        print(f"  [ROI] Scanning '{os.path.basename(video_path)}' for the "
              f"first red-light frame ...")
        frame_index = find_first_red_frame(video_path)
    print(f"  [ROI] ROI selection will use frame #{frame_index}.")

    rois = gather_rois(video_path, names, frame_index)
    save_rois_to_disk(cache_file, video_path, names, rois)
    print(f"  [ROI] Saved fresh ROIs to '{cache_file}'")
    return rois


#  RAW PEAK DIAGNOSTICS
# The functions in this section are for
# debugging / find problematic events / find problems in the videos
# by looking in more detail

def diagnose_raw_traces(traces, names, title, out_png, coinc_N=None):
    """
    Print raw statistics (max / min / mean / std) for each channel and overlay
    them on a single plot to verify saturation or misplaced ROIs before parsing.
    """
    print(f"\n  --- Raw trace diagnostics: {title} ---")
    fig, ax = plt.subplots(figsize=(10, 5))
    stats = []
    for trace, name in zip(traces, names):
        mx, mn = float(trace.max()), float(trace.min())
        mu, sd = float(trace.mean()), float(trace.std())
        stats.append((name, mx))
        print(f"    {name:10s}  max={mx:8.2f}  min={mn:7.2f}"
              f"  mean={mu:8.2f}  std={sd:8.2f}")
        ax.plot(trace, label=f"{name} (max={mx:.1f})", alpha=0.8)

    if coinc_N is not None and len(stats) == 2:
        (nA, mxA), (nB, mxB) = stats
        if mxA > 0 and mxB > 0 and (mxB < 0.2 * mxA or mxA < 0.2 * mxB):
            print(f"    WARNING: '{nA}' and '{nB}' brightness mismatch — "
                  "check ROI placement.")
        if coinc_N == 0:
            print("    WARNING: zero coincidences — check time alignment.")

    ax.set_title(f"Raw red-mean traces — {title}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean red-channel intensity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"    Diagnostic plot -> {out_png}")
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)


#  TRACE PREP: trim leading dark frames so channels align

def trim_to_common_onset(traces):
    onsets = [_find_onset_index(t, ONSET_RED_MEAN, ONSET_MAX_SEARCH) for t in traces]
    cut = min(onsets) if onsets else 0
    trimmed = [t[cut:] for t in traces]
    if any(len(t) == 0 for t in trimmed):
        return trimmed, onsets, cut
    L = min(len(t) for t in trimmed)
    return [t[:L] for t in trimmed], onsets, cut


#  PART A — density-matrix reconstruction

def coincidences_hv(event_lists, swap_bob=False):
    """
    event_lists order: [A_V, A_H, B_V, B_H]; returns (N_HH, N_HV, N_VH, N_VV).
    N_HV and N_VH are needed for the normalization.
    """
    A_V, A_H, B_V, B_H = event_lists
    if swap_bob:
        B_V, B_H = B_H, B_V
    return (count_coincidences(A_H, B_H),
            count_coincidences(A_H, B_V),
            count_coincidences(A_V, B_H),
            count_coincidences(A_V, B_V))

def build_rho(N_HH, N_HV, N_VH, N_VV, idx_pair):
    """
    Build a 4x4 density matrix (basis |HH>,|HV>,|VH>,|VV> -> indices 0,1,2,3)
    Constructs it as the outer product
    |amp><amp| with amp_i = sqrt(p_i)
    """
    counts = (N_HH, N_HV, N_VH, N_VV)
    tot = sum(counts)
    if tot == 0:
        return np.zeros((4, 4), dtype=complex)
    i, j = idx_pair
    amp = np.zeros(4, dtype=complex)
    amp[i] = np.sqrt(max(counts[i], 0) / tot)
    amp[j] = np.sqrt(max(counts[j], 0) / tot)
    return np.outer(amp, amp.conj())


def plot_rho_3d(ax, rho, title):
    """Draw a 3-D bar chart of |rho_ij| (absolute values of density matrix elements)."""
    Z = np.abs(rho)
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    x, y = xs.ravel(), ys.ravel()
    z = np.zeros_like(x, dtype=float)
    dz = Z.ravel()
    dx = dy = 0.6 * np.ones_like(dz)
    norm = mcolors.Normalize(
        vmin=float(np.min(dz)),
        vmax=float(np.max(dz)) if np.max(dz) > np.min(dz) else float(np.min(dz)) + 1e-12
    )
    cmap = cm.RdBu
    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=cmap(norm(dz)))
    ax.set_xticks(np.arange(4) + 0.3)
    ax.set_yticks(np.arange(4) + 0.3)
    ax.set_xticklabels(BASIS_LABELS,fontsize=12)
    ax.set_yticklabels(BASIS_LABELS,fontsize=12)
    ax.tick_params(axis='z', labelsize=12)
    ax.set_zlim(0, 0.55)
    ax.set_title(title, pad=10,fontsize=14)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label=r"$|\rho_{ij}|$")


def discover_part_a_videos():
    found = []
    for fn in os.listdir(PART_A_DIR):
        m = re.match(PART_A_FILENAME_RE, fn)
        if m:
            found.append((int(m.group(1)), os.path.join(PART_A_DIR, fn)))
    found.sort(key=lambda t: t[0])
    if not found:
        raise RuntimeError(f"No Part A videos in {PART_A_DIR}")
    return found

def run_part_a():
    print("\n" + "=" * 55)
    print("  PART A — quantum-state tomography")
    print("=" * 55)
    videos = discover_part_a_videos()
    print(f"  Found {len(videos)} Part A clip(s): "
          f"{[b for b, _ in videos]} bits")

    rois = acquire_rois(PART_A_ROI_FILE, videos[0][1],
                        PART_A_ROI_NAMES, RESELECT_ROI_A, ROI_FRAME_A)

    fig = plt.figure(figsize=(14, 9))
    for col, (bits, path) in enumerate(videos[:3]):
        print(f"\n  Processing {os.path.basename(path)} ({bits} bits)")
        traces, _ = measure_red_traces(path, rois)
        traces, onsets, cut = trim_to_common_onset(traces)
        if any(len(t) == 0 for t in traces):
            print("    empty trace — skipping")
            continue

        diagnose_raw_traces(
            traces, PART_A_ROI_NAMES,
            title=f"{bits} bits",
            out_png=f"diagA_raw_{bits}bits.png",
        )

        events = [trace_to_event_frames(t) for t in traces]
        n_events = [len(e) for e in events]
        print(f"    events per channel {PART_A_ROI_NAMES}: {n_events}")

        N_phi = coincidences_hv(events, swap_bob=False)
        N_psi = coincidences_hv(events, swap_bob=True)
        rho_phi = build_rho(*N_phi, idx_pair=(0, 3))
        rho_psi = build_rho(*N_psi, idx_pair=(1, 2))

        print(f"    N (HH,HV,VH,VV) no-swap = {N_phi}")
        print(f"    N (HH,HV,VH,VV) swap    = {N_psi}")
        print("    rho(|Phi+>) =\n", np.round(rho_phi, 4))
        print("    rho(|Psi+>) =\n", np.round(rho_psi, 4))

        ax1 = fig.add_subplot(2, 3, col + 1, projection="3d")
        plot_rho_3d(ax1, rho_phi, f"|Φ⁺⟩  {bits} bits")
        ax2 = fig.add_subplot(2, 3, col + 4, projection="3d")
        plot_rho_3d(ax2, rho_psi, f"|Ψ⁺⟩  {bits} bits")

    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.06,
                        wspace=0.15, hspace=0.25)
    out = "PartA_QT_density.png"
    fig.savefig(out, dpi=300)
    print(f"\n  Figure saved -> {out}")
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)


#  PART A — SIMULATION (synthetic signals)

def simulate_part_a_signals(n_shots, seed):
    """
    Generate four synthetic 1-D integer arrays (A_V, A_H, B_V, B_H) shaped
    like what 'measure_red_traces' / 'read_signals_nroi' produces from real
    video. Each pulse is a single-frame spike of height 'peak' against a
    baseline `base`. Bob is perfectly correlated with Alice in the HV basis.
    """
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, size=n_shots)   # 0 = H, 1 = V
    bob_bits   = alice_bits.copy()                  # perfectly correlated

    pre  = 60                                       # quiet leading frames
    gap  = max(MIN_EVENT_SEPARATION + 5, 30)        # frames between pulses
    T    = pre + n_shots * gap + 60                 # total signal length
    base = 50                                       # background level
    peak = max(RED_START_THRESHOLD + 500, 4000)     # pulse amplitude

    A_V = np.full(T, base, dtype=np.int64)
    A_H = np.full(T, base, dtype=np.int64)
    B_V = np.full(T, base, dtype=np.int64)
    B_H = np.full(T, base, dtype=np.int64)

    for k in range(n_shots):
        t = pre + k * gap
        if alice_bits[k] == 0:
            A_H[t] = peak           # Alice measured H
        else:
            A_V[t] = peak           # Alice measured V
        if bob_bits[k] == 0:
            B_H[t] = peak           # Bob measured H (correlated)
        else:
            B_V[t] = peak

    return [A_V, A_H, B_V, B_H]


def run_part_a_simulation(n_shots, seed):
    """
    Run the exact same Part-A pipeline (event detection -> coincidences ->
    rho) on synthetic signals whose ground truth is known. Prints per-state
    sanity-check tables and saves a 2x2 figure (ideal vs reconstructed for
    both Bell states).
    """
    print("\n" + "=" * 55)
    print(f"  PART A – SIMULATION  ({n_shots} shots, seed={seed})")
    print("=" * 55)

    sigs = simulate_part_a_signals(n_shots, seed)

    # Ground-truth bit string (regenerate exactly the same way the sim did)
    rng_truth = np.random.default_rng(seed)
    alice_bits = rng_truth.integers(0, 2, size=n_shots)
    n_zero = int(np.sum(alice_bits == 0))   # count of "0" (H) bits
    n_one  = int(np.sum(alice_bits == 1))   # count of "1" (V) bits

    # --- run the real pipeline on the synthetic signals ---
    events = [trace_to_event_frames(s) for s in sigs]

    N_phi_got = coincidences_hv(events, swap_bob=False)   # (HH, HV, VH, VV)
    N_psi_got = coincidences_hv(events, swap_bob=True)

    rho_phi_got = build_rho(*N_phi_got, idx_pair=(0, 3))
    rho_psi_got = build_rho(*N_psi_got, idx_pair=(1, 2))

    # --- ground-truth expected counts ---
    # |Phi+>:  Alice & Bob both got the same bit, no swap on Bob:
    #         bit=0 -> N_HH event,  bit=1 -> N_VV event,  no HV/VH.
    N_phi_exp = (n_zero, 0, 0, n_one)
    # |Psi+>:  with Bob V<->H swapped, bit=0 -> N_HV, bit=1 -> N_VH.
    N_psi_exp = (0, n_zero, n_one, 0)

    rho_phi_ideal = build_rho(*N_phi_exp, idx_pair=(0, 3))
    rho_psi_ideal = build_rho(*N_psi_exp, idx_pair=(1, 2))

    # --- sanity-check report ---
    def _report(state_label, expected, got):
        names = ("HH", "HV", "VH", "VV")
        print(f"\n[Sanity check {state_label}]")
        print("  Expected  " + "  ".join(f"{n}={v}" for n, v in zip(names, expected)))
        print("  Got       " + "  ".join(f"{n}={v}" for n, v in zip(names, got)))
        lost = sum(e - g for e, g in zip(expected, got))
        if expected == got:
            print("  ✓ Perfect recovery")
        else:
            print(f"  ✗ WARNING: {lost} coincidences lost / mismatched")

    _report("|Phi+>", N_phi_exp, N_phi_got)
    _report("|Psi+>", N_psi_exp, N_psi_got)

    # --- Frobenius norms ---
    frob_phi = float(np.linalg.norm(rho_phi_ideal - rho_phi_got, ord="fro"))
    frob_psi = float(np.linalg.norm(rho_psi_ideal - rho_psi_got, ord="fro"))
    print(f"\n  ||rho_ideal - rho_reconstructed||_F  (|Phi+>) = {frob_phi:.4f}")
    print(f"  ||rho_ideal - rho_reconstructed||_F  (|Psi+>) = {frob_psi:.4f}")

    # --- print the simulated (and ideal) density matrices ---
    print("\n  Ideal         rho(|Phi+>) =")
    print(np.round(rho_phi_ideal, 4))
    print("\n  Reconstructed rho(|Phi+>) =")
    print(np.round(rho_phi_got, 4))
    print("\n  Ideal         rho(|Psi+>) =")
    print(np.round(rho_psi_ideal, 4))
    print("\n  Reconstructed rho(|Psi+>) =")
    print(np.round(rho_psi_got, 4))

    # --- 2x2 figure: ideal top row, reconstructed bottom row ---
    fig = plt.figure(figsize=(11, 9))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_rho_3d(ax1, rho_phi_ideal, "Ideal |Φ⁺⟩")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_rho_3d(ax2, rho_psi_ideal, "Ideal |Ψ⁺⟩")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    plot_rho_3d(ax3, rho_phi_got, "Reconstructed |Φ⁺⟩")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    plot_rho_3d(ax4, rho_psi_got, "Reconstructed |Ψ⁺⟩")

    # fig.suptitle(
    #     f"Part A simulation verification "
    #     f"(n_shots={n_shots}, seed={seed})\n"
    #     f"||Δρ||_F  Φ+ = {frob_phi:.4f}    Ψ+ = {frob_psi:.4f}",
    #     fontsize=11,
    # )
    fig.suptitle(
        f"Part A Simulation ",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = "PartA_simulation_verification_50.png"
    fig.savefig(out, dpi=300)
    print(f"\n  Figure saved -> {out}")
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)


#  PART B — CHSH Bell inequality

def discover_part_b_videos():
    """
    Parse each file in PART_B_DIR matching PART_B_FILENAME_RE and produce a
    {(alpha, beta): path} table. Handles names like '0to22.5.mp4'.
    """
    pat = re.compile(PART_B_FILENAME_RE)
    table = {}
    for fn in os.listdir(PART_B_DIR):
        m = pat.match(fn)
        if not m:
            continue
        alpha = round_angle(m.group(1))
        beta = round_angle(m.group(2))
        table[(alpha, beta)] = os.path.join(PART_B_DIR, fn)
    if not table:
        raise RuntimeError(f"No Part B videos in {PART_B_DIR}")
    alphas = sorted({a for a, _ in table})
    betas = sorted({b for _, b in table})
    print(f"  [discover] {len(table)} Part B videos. "
          f"alphas={alphas} betas={betas}")
    return table


def find_roi_override(alpha, beta):
    """
    Look up an entry in PART_B_ROI_OVERRIDES for the given (alpha, beta) pair,
    using np.isclose so e.g. 22.5 and 22.500000001 match. Returns the override
    cache filename, or None if no override applies.
    """
    for (oa, ob), cache_path in PART_B_ROI_OVERRIDES.items():
        if np.isclose(alpha, oa) and np.isclose(beta, ob):
            return cache_path
    return None


def canon_alpha(a):
    """ Handle the polarizer angle periodicity problem."""
    a = ((a + 90.0) % 180.0) - 90.0
    if abs(a + 90.0) < 1e-9:
        a = 90.0
    return float(min(ALPHA_ANGLES, key=lambda v: abs(v - a)))


def canon_beta(b):
    """ Handle the polarizer angle periodicity problem."""
    b = b % 180.0
    if abs(b - 157.5) < 1e-6:
        return -22.5
    pool = [22.5, 67.5, 112.5, 157.5]
    pick = min(pool, key=lambda v: abs(v - b))
    return -22.5 if abs(pick - 157.5) < 1e-6 else float(pick)


def get_N(counts, a, b):
    return counts.get((canon_alpha(a), canon_beta(b)), 0)


def correlation_E(counts, a, b):
    """E = (N_ab + N_ap_bp - N_a_bp - N_ap_b) / (N_ab + N_ap_bp + N_a_bp + N_ap_b)."""
    N_ab = get_N(counts, a, b)
    N_apbp = get_N(counts, a + 90, b + 90)
    N_abp = get_N(counts, a, b + 90)
    N_apb = get_N(counts, a + 90, b)
    denom = N_ab + N_apbp + N_abp + N_apb
    if denom == 0:
        return 0.0
    return (N_ab + N_apbp - N_abp - N_apb) / denom


def chsh_S(counts):
    """S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)."""
    return sum(s * correlation_E(counts, a, b)
               for s, (a, b) in zip(CHSH_SIGNS, CHSH_PAIRS))


def sigma_E_squared(counts, a, b):
    """ E error propagation through N (Poisson)"""
    N1 = get_N(counts, a, b)
    N2 = get_N(counts, a + 90, b + 90)
    N3 = get_N(counts, a, b + 90)
    N4 = get_N(counts, a + 90, b)
    total = N1 + N2 + N3 + N4
    if total == 0:
        return 0.0
    return 4.0 * (N1 + N2) * (N3 + N4) / (total ** 3)


def sigma_S(counts):
    return float(np.sqrt(sum(sigma_E_squared(counts, a, b)
                             for a, b in CHSH_PAIRS)))


def run_part_b():
    print("\n" + "=" * 55)
    print("  PART B — CHSH Bell-inequality measurement")
    print("=" * 55)

    table = discover_part_b_videos()
    ref_video = next(iter(table.values()))
    rois = acquire_rois(PART_B_ROI_FILE, ref_video,
                        PART_B_ROI_NAMES, RESELECT_ROI_B, ROI_FRAME_B)

    counts = {}
    fig, axes = plt.subplots(len(ALPHA_ANGLES), len(BETA_ANGLES),
                             figsize=(14, 10), sharey=True)

    for i, alpha in enumerate(ALPHA_ANGLES):
        for j, beta in enumerate(BETA_ANGLES):
            key = (round_angle(alpha), round_angle(beta))
            ax = axes[i, j]
            if key not in table:
                print(f"  alpha={alpha:+6.1f}, beta={beta:+6.1f}: NO VIDEO")
                counts[(alpha, beta)] = 0
                ax.set_title(f"a={alpha}, b={beta}\n(missing)", fontsize=8)
                continue

            path = table[key]
            print(f"  {os.path.basename(path)}  "
                  f"alpha={alpha:+6.1f}, beta={beta:+6.1f} ... ",
                  end="", flush=True)

            # --- per-video ROI override (PART_B_ROI_OVERRIDES) ---
            override_cache = find_roi_override(alpha, beta)
            if override_cache is not None:
                print(f"\n  [ROI override] using '{override_cache}' "
                      f"for alpha={alpha}, beta={beta}")
                rois_for_video = acquire_rois(
                    override_cache, path, PART_B_ROI_NAMES,
                    force=False, frame_index=None,
                )
            else:
                rois_for_video = rois

            traces, _ = measure_red_traces(path, rois_for_video)
            traces, onsets, cut = trim_to_common_onset(traces)
            if any(len(t) == 0 for t in traces):
                print("empty trace, N=0")
                counts[(alpha, beta)] = 0
                continue

            events_A = trace_to_event_frames(traces[0])
            events_B = trace_to_event_frames(traces[1])
            N = count_coincidences(events_A, events_B)
            counts[(alpha, beta)] = N
            print(f"events A={len(events_A)} B={len(events_B)}, N={N}")

            nA = normalise(traces[0])
            nB = normalise(traces[1])
            ax.plot(nA, alpha=0.6, label="Alice", color='red')
            ax.plot(nB, alpha=0.6, label="Bob",  color='blue')
            ax.axhline(EVENT_THRESHOLD, ls="--", color="gray", alpha=0.6)
            ax.set_title(f"a={alpha}, b={beta}", fontsize=12)
            if i == len(ALPHA_ANGLES) - 1:
                ax.set_xlabel("Frame",fontsize=12)
            if j == 0:
                ax.set_ylabel("Intensity (Norm)", fontsize=12)

    fig.tight_layout()
    grid_png = "PartB_QT_grid.png"
    fig.savefig(grid_png, dpi=300)
    print(f"\n  Grid figure saved -> {grid_png}")
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)

    # --- CHSH math ---
    E_vals = [correlation_E(counts, a, b) for a, b in CHSH_PAIRS]
    S = chsh_S(counts)
    dS = sigma_S(counts)

    print("\n" + "-" * 48)
    print("  Correlation values:")
    for (a, b), s, e in zip(CHSH_PAIRS, CHSH_SIGNS, E_vals):
        sign = "+" if s > 0 else "-"
        print(f"    {sign} E(alpha={a:>5}, beta={b:>5}) = {e:+.4f}")
    print(f"\n  CHSH parameter   S = {S:+.4f}  +/-  {dS:.4f}")
    if abs(S) > 2:
        print(f"  |S| > 2  ->  classical (local hidden-variable) bound VIOLATED")
        print(f"  Quantum prediction for a Bell state: 2*sqrt(2) "
              f"= {2.0 * np.sqrt(2):.4f}")
    else:
        print(f"  |S| <= 2  ->  classical bound NOT violated.")
    print("-" * 48)


#  ENTRY POINT

def main():
    print("=" * 55)
    print(f"  QT_and_Bell_Test.py    {timestamp()}")
    print("=" * 55)
    print(f"  DO_PART_A        = {DO_PART_A}")
    print(f"  DO_PART_B        = {DO_PART_B}")
    print(f"  RUN_SIMULATION_A = {RUN_SIMULATION_A}")
    print(f"  EVENT_THRESHOLD    = {EVENT_THRESHOLD}")
    print(f"  COINCIDENCE_WINDOW = {COINCIDENCE_WINDOW}")

    if DO_PART_A:
        run_part_a()
    if RUN_SIMULATION_A:
        run_part_a_simulation(n_shots=SIM_A_SHOTS, seed=SIM_A_SEED)
    if DO_PART_B:
        run_part_b()
    print("\n  Done.")


if __name__ == "__main__":
    main()
