# Enhanced Plagiarism Detection - v2.0

## Summary of Improvements

This update significantly reduces **false positives** in plagiarism detection by implementing:

### 1. Semantic-Lexical Divergence Check ⭐ (KEY IMPROVEMENT)

**Problem Solved**: Two papers discussing the *same topic* with *completely different wording* were being flagged as plagiarism.

**Solution**: The system now compares:
- **Semantic Score**: From embedding similarity (Qdrant vector search)
- **Lexical Score**: From word/n-gram overlap

If semantic is HIGH but lexical is LOW → **NOT plagiarism** (just same topic)

```python
# Example: These discuss the same topic but aren't plagiarism
Text A: "Machine learning has revolutionized medical imaging..."
Text B: "AI techniques are transforming how doctors analyze scans..."

Semantic: 0.85 (high - same topic)
Lexical: 0.10 (low - different words)
→ Flagged as COMMON_TOPIC, not plagiarism!
```

### 2. Multi-Scale N-gram Analysis

Instead of fixed 5-gram, now uses multiple scales: **[3, 4, 5, 7]-grams**

- Catches paraphrases that keep 2-3 word phrases
- Better detection of mosaic plagiarism
- Takes the best score across all scales

### 3. Citation Context Detection

Automatically detects properly cited text:
- `(Smith et al., 2023)`
- `[1,2,3]`
- `according to previous studies`
- `as demonstrated by`

Cited content gets **30% score reduction** and may be classified as `PROPER_CITATION`.

### 4. Confidence Scoring

Every match now includes a **confidence score** (0-1):
- High confidence: Likely real plagiarism
- Low confidence: May be false positive

Matches below `min_confidence` threshold (default 0.3) are filtered out.

### 5. Sentence-Level Analysis

For promising matches (score > 0.4), the system:
1. Breaks text into sentences
2. Compares each sentence individually
3. Reports which specific sentences match
4. Identifies if matched sentences are cited

### 6. Enhanced Boilerplate Filtering

Better detection of common scientific phrases that shouldn't trigger plagiarism:
- Methods sections: "statistical analysis was performed"
- Results: "p < 0.05 was considered significant"
- Standard formulations: "data are presented as mean ± SD"

---

## New Match Types

| Type | Description |
|------|-------------|
| `EXACT_COPY` | Near-identical text (>95% lexical) |
| `DIRECT_COPY` | High similarity (>80% lexical) |
| `PARAPHRASE` | Rewording with similar meaning |
| `MOSAIC` | Mixed original + copied sentences |
| `COMMON_TOPIC` | **NEW**: Same subject, different wording |
| `COMMON_KNOWLEDGE` | Generic scientific knowledge |
| `PROPER_CITATION` | **NEW**: Text appears to be properly cited |
| `NO_MATCH` | Insufficient similarity |

---

## Configuration Options

```python
DetectorConfig(
    # Standard thresholds
    text_exact_threshold=0.95,
    text_high_threshold=0.80,
    text_moderate_threshold=0.50,
    
    # NEW: Enhanced detection options
    use_enhanced_detection=True,     # Enable new features
    divergence_threshold=0.45,       # Semantic-lexical gap threshold
    citation_boost=0.3,              # Score reduction for cited text
    min_confidence=0.3,              # Minimum confidence to report
)
```

---

## Test Results

```
✓ Test 1 (Exact Copy): PASSED
✓ Test 2 (False Positive Prevention): PASSED  ← Key improvement!
✓ Test 3 (Citation Detection): PASSED
✓ Test 4 (Paraphrase Detection): PASSED
✓ Test 5 (Unrelated Text): PASSED
✓ Test 6 (Boilerplate Handling): PASSED
✓ Test 7 (Mosaic Detection): PASSED

TOTAL: 7/7 tests passed
```

---

## Files Changed

1. **`detection/text_detector_enhanced.py`** (NEW)
   - EnhancedTextPlagiarismDetector class
   - All new detection logic

2. **`detection/world_class_detector.py`** (MODIFIED)
   - Updated to use enhanced detector
   - Passes semantic scores from Qdrant
   - New filtering logic

3. **`detection/__init__.py`** (MODIFIED)
   - Exports new classes
   - Falls back to standard detector if needed

4. **`test_enhanced_detector.py`** (NEW)
   - Comprehensive test suite
   - 7 test cases for all scenarios

---

## Backward Compatibility

The enhanced detector is **100% backward compatible**:

```python
# Old code still works
from detection import TextPlagiarismDetector
detector = TextPlagiarismDetector()
result = detector.compare(query, source)

# result.similarity_score  ← still works
# result.match_type        ← still works
# result.containment       ← still works
```

New fields are available on the result object:
- `result.confidence`
- `result.semantic_score`
- `result.lexical_score`
- `result.divergence_flag`
- `result.sentence_matches`
- `result.has_citation_context`

---

## Running Tests

```bash
cd uyari_updated
python test_enhanced_detector.py
```

---

## Usage Example

```python
from detection import EnhancedTextPlagiarismDetector

detector = EnhancedTextPlagiarismDetector(
    divergence_threshold=0.45,
    citation_boost=0.3
)

# With semantic score from Qdrant search
result = detector.compare(
    query_text="...",
    source_text="...",
    semantic_score=0.85  # From Qdrant cosine similarity
)

# Check if it's a false positive
if result.divergence_flag:
    print("Same topic, different wording - NOT plagiarism")

# Check confidence
if result.confidence < 0.5:
    print("Low confidence match - review manually")

# Check citation
if result.has_citation_context:
    print(f"Text appears cited ({result.citation_type})")
```
