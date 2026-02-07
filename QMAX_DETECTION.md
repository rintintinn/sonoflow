# Qmax Detection Algorithm

This document describes the methodology for maximum flow rate (Qmax) detection in acoustic uroflowmetry.

## Problem Statement

Acoustic uroflowmetry converts sound energy to flow rate, but several factors can cause **Qmax overestimation**:

1. **Transient onset spikes** - Initial jet formation and splash turbulence at voiding start
2. **High-slope regions** - Rapid flow rate changes during unstable flow
3. **Short-duration peaks** - Brief acoustic artifacts that don't represent sustained flow

**Goal**: Measure Qmax that represents sustained, stable flow - not transient spikes.

---

## Qmax Methods Implemented

We calculate **6 different Qmax values** for comparison and validation:

| Method | Description | Use Case |
|--------|-------------|----------|
| `qmax` | Max of minimal-smoothed flow | Most sensitive, closest to raw |
| `qmax_smoothed` | Max of EMA-smoothed flow | Physiological envelope |
| `qmax_ics_sliding` | Sliding window avg (300ms) | ICS compliance |
| `qmax_ics_consecutive` | Sustained max (300ms consec) | Strictest ICS |
| `qmax_slope_stabilized` | Max in stable-slope regions | Debug/research only |

---

## Method 1: Minimal Smoothing (Default Display)

```python
# Step 1: Apply short median filter to suppress transient spikes
flow_minimal = median_filter(flow_rate, size=3)

# Step 2: Apply 0.5s onset exclusion window
exclusion_frames = int(0.5 / hop_length_sec)  # ~20 frames
flow_for_qmax = flow_minimal[exclusion_frames:]

# Step 3: Take maximum
qmax = max(flow_for_qmax)
```

**Parameters**:
- Median filter: 3 frames (~75ms at 25ms hop)
- Onset exclusion: 0.5 seconds from voiding start

---

## Method 2: Full Smoothing

```python
# After median filter, apply causal EMA
ema_alpha = 0.15
flow_smooth = causal_ema(flow_minimal, alpha=ema_alpha)
qmax_smoothed = max(flow_smooth)
```

**Why causal EMA**: Produces physiologically plausible flow envelope; non-causal smoothing would "look ahead" and distort the curve shape.

---

## Method 3: ICS Sliding Window (300ms)

Per International Continence Society guidelines, Qmax should be sustained, not a transient spike.

```python
ICS_WINDOW_FRAMES = 12  # ~300ms at 25ms hop

# Compute sliding window average
kernel = np.ones(ICS_WINDOW_FRAMES) / ICS_WINDOW_FRAMES
sliding_avg = np.convolve(flow_rate, kernel, mode='valid')

qmax_ics_sliding = max(sliding_avg)
```

**Rationale**: Takes the maximum of 300ms running averages, ensuring Qmax represents at least 300ms of sustained flow.

---

## Method 4: ICS Consecutive Frames (300ms)

More strict interpretation of ICS guidelines using binary search.

```python
# Binary search for highest value sustained ≥300ms
peak = max(flow_rate)
min_val, max_val = 0.0, peak

for _ in range(20):  # Precision iterations
    mid = (min_val + max_val) / 2
    
    # Find longest run of consecutive frames ≥ mid
    longest_run = find_longest_run(flow_rate >= mid)
    
    if longest_run >= ICS_WINDOW_FRAMES:
        min_val = mid  # Achievable
    else:
        max_val = mid  # Too high

qmax_ics_consecutive = min_val
```

**Difference from sliding**: Requires ALL frames in window to exceed threshold, not just average.

---

## Method 5: Slope-Stabilized Qmax (Debug Only)

**Problem**: Even with onset exclusion, high dQ/dt regions during voiding can inflate Qmax.

### Algorithm

```python
# Step 1: Compute first-order difference
dQ_dt = np.diff(flow_rate) / dt  # dt = hop_length_sec

# Step 2: Adaptive slope threshold (90th percentile)
# EXCLUDE first 300ms from threshold calibration
slope_exclusion_frames = int(0.3 / hop_length_sec)
abs_dQ_dt_for_threshold = abs(dQ_dt)[slope_exclusion_frames:]
slope_threshold = np.percentile(abs_dQ_dt_for_threshold, 90)

# Step 3: Mark stable regions (|dQ/dt| ≤ threshold for ≥200ms)
stable_mask = find_stable_regions(abs(dQ_dt), slope_threshold, min_frames=8)

# Step 4: Find max sustained Qmax in stable regions only (≥300ms)
qmax_slope_stabilized = find_sustained_max_in_stable(flow_rate, stable_mask)
```

**Parameters**:
- Slope percentile: 90% (data-adaptive, not fixed)
- Threshold calibration exclusion: 300ms (prevents onset turbulence inflation)
- Stable region minimum: 200ms (8 frames)
- Qmax sustained requirement: 300ms (12 frames)
- Qmax eligibility window: ≥0.5s after onset (unchanged)

### Test Results

| Recording | Qmax Standard | Qmax Slope-Stabilized | Reduction |
|-----------|---------------|----------------------|-----------|
| `2026_02_07_12_43_34_1.wav` | 29.5 ml/s | 14.9 ml/s | -14.6 ml/s (50%) |

---

## Onset Exclusion Window

All Qmax methods exclude the first 0.5 seconds after voiding onset.

```python
QMAX_ONSET_EXCLUSION_SEC = 0.5  # seconds

# At 25ms hop length:
exclusion_frames = int(0.5 / 0.025)  # = 20 frames
```

**Why**: Initial jet formation and splash turbulence create artificially high acoustic energy that does not represent steady-state flow rate.

---

## Volume Normalization

Before Qmax calculation, flow rate is normalized to match user-provided voided volume:

```python
# Compute area under flow proxy curve
area = np.trapz(flow_proxy, time_axis)

# Scale to actual volume
scale_factor = volume_ml / area
flow_rate = flow_proxy * scale_factor
```

This ensures Qmax is expressed in ml/s regardless of microphone sensitivity or distance.

---

## Key Learnings

1. **Median filter before max** suppresses transient acoustic spikes without distorting flow shape
2. **Onset exclusion (0.5s)** prevents jet/splash artifacts from inflating Qmax
3. **ICS 300ms window** ensures Qmax represents sustained flow, not brief peaks
4. **Slope-based filtering** (debug) reveals that standard Qmax may be 2x actual sustained flow
5. **Adaptive thresholds** (percentile-based) work better than fixed constants across recordings

---

## Recommendations

| Use Case | Recommended Method |
|----------|-------------------|
| Clinical display | `qmax` (minimal smoothing + onset exclusion) |
| ICS compliance | `qmax_ics_sliding` or `qmax_ics_consecutive` |
| Research/validation | Compare all 5 methods |
| Strict analysis | `qmax_slope_stabilized` (requires debug mode) |
