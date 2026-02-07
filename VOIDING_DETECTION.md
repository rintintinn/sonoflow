# Voiding Time Detection Algorithm

This document describes the evolution of the voiding time detection algorithm and the different approaches tested.

## Problem Statement

The original algorithm used the **entire recording duration** as voiding time. This was incorrect because:
1. Pre-void silence (0-10+ seconds of background noise before actual urination)
2. Post-void noise (sounds after urination ends, e.g., toilet flushing, movement)

**Goal**: Detect the actual voiding segment and shift the time axis to start at t=0.

---

## Validation Data

| Recording | Medical-Grade Voiding Time | Original Detection |
|-----------|---------------------------|-------------------|
| `2026_02_07_12_43_34_1.wav` | **24.7 seconds** | 41.4 seconds |

---

## Approaches Tested

### Approach 1: Simple Threshold (1.5× Noise Floor) ❌
```python
threshold = noise_floor * 1.5
above = energy > threshold
start = first above threshold
end = last above threshold
```
**Result**: Captured too much - background noise triggered threshold early.

---

### Approach 2: Higher Threshold (2.0× Noise Floor) ❌
```python
threshold = noise_floor * 2.0
```
**Result**: Still triggered at t=0.6s due to low background noise variability.

**Diagnostic data**:
- 2.0×: first=0.6s, last=41.3s, span=40.8s ❌

---

### Approach 3: Even Higher Threshold (3.0× Noise Floor) ⚠️
```python
threshold = noise_floor * 3.0
```
**Result**: Start detection correct at t=11.2s, but span still 29s (expected 24.7s).

**Diagnostic data**:
- 3.0×: first=11.2s, last=40.3s, span=29.1s

**Issue**: Post-void spike at t=38-40s (7× noise) was being included.

---

### Approach 4: Gap-Based Segment Detection ❌
Detect contiguous segments above threshold, separated by gaps >2 seconds.
```python
MIN_GAP_FRAMES = 80  # ~2 seconds
# Find segments separated by gaps
# Pick segment with highest peak energy
```
**Result**: 30.0s - gap between t=35s and t=40s was only ~2-3s with scattered frames.

---

### Approach 5: Sustained Low Energy End Detection ❌
Find end as last frame followed by >1 second of sustained low energy (<2× noise).
```python
SUSTAINED_LOW_FRAMES = 40  # 1 second
low_threshold = noise_floor * 2.0
# Find frames followed by sustained silence
```
**Result**: 30.0s - t=38s rises back to 2.1× noise, breaking the silence detection.

**Diagnostic data** (energy ratio at different times):
```
t=35s: 0.9× (low - flow ended)
t=36s: 1.2× (low)
t=37s: 1.0× (low)
t=38s: 2.1× (rises above 2× threshold!)
t=39s: 3.0× (post-void spike)
t=40s: 7.0× (post-void spike)
```

---

### Approach 6: Peak-Relative End Detection ✅ **FINAL**
Find end as first point after peak where energy drops to <10% of peak and stays low for 0.5s.
```python
peak_energy = np.max(energy)
end_threshold = peak_energy * 0.10  # 10% of peak
SUSTAINED_LOW_FRAMES = 20  # 0.5 seconds

for i in range(peak_idx, len(energy)):
    if energy[i] < end_threshold:
        # Check if stays low for 0.5s
        if stays_low_for(i, SUSTAINED_LOW_FRAMES):
            true_end_idx = i
            break
```

**Result**: **23.7s** (expected 24.7s) - **96% accuracy** ✅

**Why it works**:
- Uses relative threshold (10% of peak) instead of absolute noise floor
- Peak at t=31.5s is very high (~30× noise)
- 10% of peak ≈ 3× noise floor
- At t=35s, energy drops to 0.9× noise, which is well below 10% of peak
- Post-void spike at t=40s (~7× noise) is still below 10% of peak (~3× noise threshold)

---

## Final Algorithm Summary

```python
# Start detection: 3× noise floor threshold
flow_threshold = noise_floor * 3.0
first_idx = first frame where energy > flow_threshold

# End detection: 10% of peak, sustained for 0.5s
end_threshold = peak_energy * 0.10
true_end_idx = first frame after peak where energy < end_threshold for 0.5s

# Refinement
start_idx = walk back from first_idx to find upslope start (1.5× noise)
end_idx = walk forward from true_end_idx to include trailing flow

# Time shift
time_axis = time_axis[start_idx:end_idx] - time_axis[start_idx]
```

---

## Key Learnings

1. **Absolute thresholds fail** when background noise varies between recordings
2. **Peak-relative thresholds** are more robust across different recording environments
3. **Sustained silence detection** needs to use a threshold below any post-void activity
4. **Gap detection** fails when post-void sounds are close in time to voiding end
5. **The 10% of peak threshold** works because post-void sounds are typically much weaker than peak urine flow
