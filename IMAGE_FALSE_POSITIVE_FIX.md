# Image False Positive Fix (v13)

## Problem
The plagiarism detector was returning false positives for images. When checking a paper:
- CNN similarity finds visually similar images (90%+ match)
- But these are often similar-looking charts/graphs from DIFFERENT papers
- Example: Two bar charts look similar (90% CNN) but are completely different (0% hash)

## Root Cause
The system used only CNN (neural network) similarity for image matching. CNN captures:
- Color patterns
- Visual texture
- Overall appearance

But CNN CANNOT distinguish:
- Different bar charts with same color scheme
- Similar scatter plots from different experiments
- Generic scientific figure templates

## Solution
Added **perceptual hash validation** as a second check:

```python
# Old logic (v12 and earlier):
if cnn_score >= 0.90:
    return MATCH  # Many false positives!

# New logic (v13):
if cnn_score >= 0.98:
    return MATCH  # Very high - likely exact copy
elif cnn_score >= 0.90 AND phash_similarity >= 0.15:
    return MATCH  # Both visual AND structural match
else:
    return REJECT  # CNN high but structure different = false positive
```

## How It Works

### Perceptual Hash (pHash)
- Captures the **structural layout** of an image
- Based on DCT (Discrete Cosine Transform)
- Measures edge patterns, shapes, spatial arrangement
- Two identical images → hash similarity ~100%
- Two different charts → hash similarity ~0%

### Combined Detection
| CNN Score | Hash Score | Result | Reason |
|-----------|------------|--------|--------|
| 92% | 0% | ❌ REJECT | Similar appearance, different structure |
| 92% | 25% | ✅ ACCEPT | Both visual and structural match |
| 99% | 0% | ✅ ACCEPT | Very high CNN = likely exact copy |
| 85% | 50% | ❌ REJECT | CNN too low |

## Test Results

Before fix:
- 49 image matches from OTHER papers
- All showing 75-92% CNN similarity
- All with 0% hash similarity (hamming distance 61-64)

After fix:
- 0 false positive matches
- Only true matches (CNN + hash both high) are reported

## Configuration

In `world_class_detector.py`:
```python
# Line 766-772
MIN_SCORE_THRESHOLD = 0.90      # Minimum CNN score to consider
REQUIRE_HASH_CONFIRMATION = True # Enable hash validation
MIN_HASH_SIMILARITY = 0.15      # Minimum hash similarity required
```

## Files Changed
- `detection/world_class_detector.py` - Added hash validation logic

## Version
- v13: Hash validation fix for image false positives
