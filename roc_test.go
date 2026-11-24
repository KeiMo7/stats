package stats_test

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/montanaflynn/stats"
)

// refExactAUC computes the weighted Mann-Whitney U statistic with exact
// rational arithmetic: each positive/negative pair contributes 1 when the
// positive outranks the negative, 1/2 on a tie and 0 otherwise. It is the
// independent ground truth the streaming accumulator is checked against.
// ok is false when either class has zero total weight.
func refExactAUC(scores []float64, labels []int, weights []float64) (float64, bool) {
	num := new(big.Rat)
	wp, wn := new(big.Rat), new(big.Rat)
	addRat := func(r *big.Rat, x float64) {
		f := new(big.Rat).SetFloat64(x)
		r.Add(r, f)
	}
	for i := range scores {
		if labels[i] != 0 && labels[i] != 1 {
			continue
		}
		if math.IsNaN(scores[i]) || math.IsInf(scores[i], 0) {
			continue
		}
		if labels[i] == 1 {
			addRat(wp, weightAt(weights, i))
		} else {
			addRat(wn, weightAt(weights, i))
		}
	}
	// Full pairwise pass: every distinct positive/negative pair contributes
	// 1, 1/2 or 0 according to its score ordering.
	for i := 0; i < len(scores); i++ {
		if labels[i] != 1 || math.IsNaN(scores[i]) || math.IsInf(scores[i], 0) {
			continue
		}
		for j := 0; j < len(scores); j++ {
			if labels[j] != 0 || math.IsNaN(scores[j]) || math.IsInf(scores[j], 0) {
				continue
			}
			credit := new(big.Rat)
			switch {
			case scores[i] > scores[j]:
				credit.SetFrac64(1, 1)
			case scores[i] < scores[j]:
				credit.SetFrac64(0, 1)
			default:
				credit.SetFrac64(1, 2)
			}
			pw := new(big.Rat).SetFloat64(weightAt(weights, i))
			nw := new(big.Rat).SetFloat64(weightAt(weights, j))
			num.Add(num, new(big.Rat).Mul(new(big.Rat).Mul(pw, nw), credit))
		}
	}
	if wp.Sign() == 0 || wn.Sign() == 0 {
		return math.NaN(), false
	}
	auc := new(big.Rat).Quo(num, new(big.Rat).Mul(wp, wn))
	f, _ := auc.Float64()
	return f, true
}

func refMetrics(scores []float64, labels []int, weights []float64, threshold float64) (tp, fp, totalPos float64) {
	for i := range scores {
		if labels[i] != 0 && labels[i] != 1 {
			continue
		}
		if math.IsNaN(scores[i]) || math.IsInf(scores[i], 0) {
			continue
		}
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if math.IsNaN(w) {
			w = 0
		}
		if labels[i] == 1 {
			totalPos += w
			if scores[i] >= threshold {
				tp += w
			}
		} else if scores[i] >= threshold {
			fp += w
		}
	}
	return tp, fp, totalPos
}

// weightAt returns the weight of sample i, defaulting to 1 and treating NaN
// weights as zero, matching the accumulator's conventions.
func weightAt(weights []float64, i int) float64 {
	if weights == nil {
		return 1
	}
	if math.IsNaN(weights[i]) {
		return 0
	}
	return weights[i]
}

type aucCase struct {
	name    string
	scores  []float64
	labels  []int
	weights []float64
}

func TestExactAUCTable(t *testing.T) {
	cases := []aucCase{
		{
			name:   "perfect separation",
			scores: []float64{0.1, 0.2, 0.8, 0.9},
			labels: []int{0, 0, 1, 1},
		},
		{
			name:   "complete inversion",
			scores: []float64{0.9, 0.8, 0.2, 0.1},
			labels: []int{0, 0, 1, 1},
		},
		{
			name:   "all scores tied",
			scores: []float64{2, 2, 2, 2},
			labels: []int{1, 0, 1, 0},
		},
		{
			name:   "tie block with mixed classes",
			scores: []float64{1, 2, 2, 2, 3},
			labels: []int{0, 1, 0, 1, 1},
		},
		{
			name:   "imbalanced one positive among many negatives",
			scores: []float64{0.5, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49},
			labels: []int{1, 0, 0, 0, 0, 0, 0},
		},
		{
			name:    "decimal weights",
			scores:  []float64{0.1, 0.4, 0.35, 0.8},
			labels:  []int{0, 1, 0, 1},
			weights: []float64{0.1, 0.25, 1.5, 0.7},
		},
		{
			name:    "zero and missing weights",
			scores:  []float64{0.1, 0.4, 0.35, 0.8},
			labels:  []int{0, 1, 0, 1},
			weights: []float64{0, math.NaN(), 1.5, 0.7},
		},
		{
			name:   "missing labels and non-finite scores are skipped",
			scores: []float64{0.1, math.NaN(), 0.4, math.Inf(1), math.Inf(-1), 0.8},
			labels: []int{0, 1, 1, 1, 0, 1},
		},
		{
			name:   "single element",
			scores: []float64{0.5},
			labels: []int{1},
		},
		{
			name:   "only negatives",
			scores: []float64{0.1, 0.2, 0.3},
			labels: []int{0, 0, 0},
		},
		{
			name:   "only positives",
			scores: []float64{0.1, 0.2, 0.3},
			labels: []int{1, 1, 1},
		},
		{
			name:    "all weights zero",
			scores:  []float64{0.1, 0.9},
			labels:  []int{0, 1},
			weights: []float64{0, 0},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := stats.NewExactAUC()
			if err := a.AddBatch(stats.Float64Data(c.scores), c.labels, stats.Float64Data(c.weights)); err != nil {
				t.Fatalf("AddBatch: %v", err)
			}
			want, ok := refExactAUC(c.scores, c.labels, c.weights)
			got, err := a.AUC()
			if !ok {
				if err != stats.ErrUndefined || !math.IsNaN(got) {
					t.Fatalf("AUC = %v, %v; want NaN, ErrUndefined", got, err)
				}
				if pts, err := a.ROCCurve(); err != stats.ErrUndefined || pts != nil {
					t.Fatalf("ROCCurve = %v, %v; want nil, ErrUndefined", pts, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("AUC error: %v", err)
			}
			if got != want {
				t.Errorf("AUC = %.17g, want exact %.17g", got, want)
			}
			// The curve endpoints must span the full unit square and the
			// trapezoidal area of the returned curve must equal AUC.
			pts, err := a.ROCCurve()
			if err != nil {
				t.Fatalf("ROCCurve: %v", err)
			}
			first, last := pts[0], pts[len(pts)-1]
			if first.FPR != 0 || first.TPR != 0 || !math.IsNaN(first.Precision) {
				t.Errorf("first point = %+v; want FPR/TPR 0 with NaN precision", first)
			}
			// The last point classifies everything positive: TPR and FPR
			// both reach 1 and precision equals the positive base rate.
			if last.FPR != 1 || last.TPR != 1 {
				t.Errorf("last point = %+v; want FPR/TPR 1", last)
			}
			// Recomputing the area from the returned float64 points uses a
			// different operation order than the exact rational AUC, so
			// allow a single ULP; the exact value itself is checked below.
			area := 0.0
			for i := 1; i < len(pts); i++ {
				area += (pts[i].FPR - pts[i-1].FPR) * (pts[i].TPR + pts[i-1].TPR) / 2
			}
			if !ulpClose(area, got, 1) {
				t.Errorf("curve area %.17g != AUC %.17g by more than one ULP", area, got)
			}
		})
	}
}

func TestExactAUCMetricsAt(t *testing.T) {
	scores := []float64{1, 2, 2, 2, 3}
	labels := []int{0, 1, 0, 1, 1}
	a := stats.NewExactAUC()
	if err := a.AddBatch(stats.Float64Data(scores), labels, nil); err != nil {
		t.Fatal(err)
	}

	// Threshold inside the tie block at score 2: the whole block moves
	// together (all-or-nothing tie rule), there is no split.
	for _, thr := range []float64{2.0, 1.5, 2.5 - 0.5} {
		precision, recall, f1, tp, fp, err := a.MetricsAt(thr)
		if err != nil {
			t.Fatalf("thr %v: %v", thr, err)
		}
		rtp, rfp, rtot := refMetrics(scores, labels, nil, thr)
		if tp != rtp || fp != rfp {
			t.Errorf("thr %v: tp/fp = %v/%v, want %v/%v", thr, tp, fp, rtp, rfp)
		}
		wantPrecision := rtp / (rtp + rfp)
		wantRecall := rtp / rtot
		wantF1 := 2 * wantPrecision * wantRecall / (wantPrecision + wantRecall)
		if precision != wantPrecision || recall != wantRecall || f1 != wantF1 {
			t.Errorf("thr %v: metrics = %.4f/%.4f/%.4f, want %.4f/%.4f/%.4f",
				thr, precision, recall, f1, wantPrecision, wantRecall, wantF1)
		}
	}

	// Threshold above every score: nothing predicted positive, precision
	// and F1 are NaN rather than a misleading zero.
	precision, recall, f1, tp, fp, err := a.MetricsAt(4)
	if err != nil || tp != 0 || fp != 0 || recall != 0 ||
		!math.IsNaN(precision) || !math.IsNaN(f1) {
		t.Errorf("threshold above max: %v %v %v %v %v %v", precision, recall, f1, tp, fp, err)
	}

	// A single-class accumulator cannot define precision-based metrics.
	one := stats.NewExactAUC()
	if err := one.Add(0.5, 0, 1); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, _, err := one.MetricsAt(0); err != stats.ErrUndefined {
		t.Errorf("single-class MetricsAt err = %v, want ErrUndefined", err)
	}
}

func TestHistogramMetricsAtThresholds(t *testing.T) {
	h, err := stats.NewHistogramAUC(stats.ScoreAxis{Min: 0, Max: 1, Bins: 4})
	if err != nil {
		t.Fatal(err)
	}
	// One positive and one negative in the lowest bin [0, 0.25).
	if err := h.Add(0.1, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := h.Add(0.2, 0, 1); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		threshold float64
		wantTP    float64
		wantFP    float64
	}{
		{-1, 1, 1},   // below the axis: everything selected
		{0, 1, 1},    // at Min: all bins selected
		{0.25, 0, 0}, // first bin's upper edge: lowest bin excluded
		{1, 0, 0},    // at Max: nothing selected
		{5, 0, 0},    // above Max: nothing selected, no panic
	}
	for _, c := range cases {
		_, _, _, tp, fp, mErr := h.MetricsAt(c.threshold)
		if mErr != nil {
			t.Errorf("threshold %v: unexpected error %v", c.threshold, mErr)
			continue
		}
		if tp != c.wantTP || fp != c.wantFP {
			t.Errorf("threshold %v: tp/fp = %v/%v, want %v/%v",
				c.threshold, tp, fp, c.wantTP, c.wantFP)
		}
	}
}

func TestExactAUCRejectsNegativeWeight(t *testing.T) {
	a := stats.NewExactAUC()
	if err := a.Add(0.5, 1, 1); err != nil {
		t.Fatal(err)
	}
	err := a.Add(0.6, 1, -0.5)
	if err != stats.ErrNegative {
		t.Fatalf("err = %v, want ErrNegative", err)
	}
	err = a.AddBatch(
		stats.Float64Data{0.2, 0.3},
		[]int{0, 1},
		stats.Float64Data{1, -1},
	)
	if err != stats.ErrNegative {
		t.Fatalf("batch err = %v, want ErrNegative", err)
	}
	// State must be untouched by the rejected calls.
	pos, neg := a.TotalWeight()
	if pos != 1 || neg != 0 || a.Len() != 1 {
		t.Fatalf("state polluted: pos=%v neg=%v len=%d", pos, neg, a.Len())
	}
	if _, err := a.AUC(); err != stats.ErrUndefined {
		t.Errorf("AUC after only positives: err = %v, want ErrUndefined", err)
	}
}

func TestExactAUCSizeMismatch(t *testing.T) {
	a := stats.NewExactAUC()
	if err := a.AddBatch(stats.Float64Data{0.1, 0.2}, []int{0}, nil); err != stats.ErrSize {
		t.Errorf("labels length: err = %v, want ErrSize", err)
	}
	if err := a.AddBatch(
		stats.Float64Data{0.1, 0.2},
		[]int{0, 1},
		stats.Float64Data{1},
	); err != stats.ErrSize {
		t.Errorf("weights length: err = %v, want ErrSize", err)
	}
	if err := a.Merge(nil); err != stats.ErrSize {
		t.Errorf("Merge(nil): err = %v, want ErrSize", err)
	}
}

func TestExactAUCBatchAtomicityAndSkipping(t *testing.T) {
	a := stats.NewExactAUC()
	good := stats.Float64Data{0.1, 0.9}
	if err := a.AddBatch(good, []int{0, 1}, nil); err != nil {
		t.Fatal(err)
	}
	posBefore, negBefore := a.TotalWeight()

	// A batch carrying a negative weight anywhere must be rejected wholesale.
	err := a.AddBatch(
		stats.Float64Data{0.2, 0.3, 0.4},
		[]int{1, 0, 1},
		stats.Float64Data{1, -2, 1},
	)
	if err != stats.ErrNegative {
		t.Fatalf("err = %v, want ErrNegative", err)
	}
	posAfter, negAfter := a.TotalWeight()
	if posAfter != posBefore || negAfter != negBefore || a.Len() != 2 {
		t.Fatalf("rejected batch polluted state: %v/%v len=%d", posAfter, negAfter, a.Len())
	}

	// Missing labels, non-finite scores and NaN weights are skipped, not
	// errors, and are counted so callers can monitor data quality.
	if err := a.Add(math.NaN(), 1, 1); err != nil {
		t.Errorf("NaN score errored: %v", err)
	}
	if err := a.Add(math.Inf(-1), 0, 1); err != nil {
		t.Errorf("-Inf score errored: %v", err)
	}
	if err := a.Add(0.5, 9, 1); err != nil {
		t.Errorf("missing label errored: %v", err)
	}
	if err := a.Add(0.6, 1, math.NaN()); err != nil {
		t.Errorf("NaN weight errored: %v", err)
	}
	if a.Skipped() != 3 {
		t.Errorf("Skipped = %d, want 3 (NaN-weight sample is valid but zero-weight)", a.Skipped())
	}
	// The NaN-weight sample added a score block with no total weight.
	pos2, neg2 := a.TotalWeight()
	if pos2 != posBefore || neg2 != negBefore {
		t.Errorf("zero-weight sample changed totals: %v/%v vs %v/%v", pos2, neg2, posBefore, negBefore)
	}
}

func indexOrder(n int, reverse bool) []int {
	idx := make([]int, n)
	for i := range idx {
		if reverse {
			idx[i] = n - 1 - i
		} else {
			idx[i] = i
		}
	}
	return idx
}

func shuffledOrder(n, seed int) []int {
	rng := rand.New(rand.NewSource(int64(seed)))
	idx := indexOrder(n, false)
	rng.Shuffle(n, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	return idx
}

// TestExactAUCDeterminism feeds one fixed dataset through three shardings and
// three arrival orders (9 configurations) and asserts every reported value is
// bit-for-bit identical, including after state merges.
func TestExactAUCDeterminism(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const n = 2400
	scores := make([]float64, n)
	labels := make([]int, n)
	weights := make([]float64, n)
	for i := 0; i < n; i++ {
		// Quantized score space so the number of blocks stays bounded.
		scores[i] = math.Floor(rng.Float64()*50) / 10
		// Clean binary labels with a ~1/1000 positive base rate, plus a
		// guaranteed set of positives so both classes are represented.
		if i < 30 || rng.Intn(1000) == 0 {
			labels[i] = 1
		}
		weights[i] = float64(rng.Intn(100)+1) / 10
	}

	orderings := map[string][]int{
		"forward":  indexOrder(n, false),
		"reverse":  indexOrder(n, true),
		"shuffled": shuffledOrder(n, 116),
	}
	shardings := []int{1, 8, 16}
	thresholds := []float64{-1, 0.05, 1.0, 2.5, 4.95, 100}

	var refAUC float64
	var refCurve []stats.ROCPoint
	var refPos, refNeg float64
	var refMetricsSnap []metricsSnapshot
	first := true
	for _, shardCount := range shardings {
		for orderName, order := range orderings {
			accs := make([]*stats.ExactAUC, shardCount)
			for k := range accs {
				accs[k] = stats.NewExactAUC()
			}
			for _, i := range order {
				if err := accs[i%shardCount].Add(scores[i], labels[i], weights[i]); err != nil {
					t.Fatal(err)
				}
			}
			merged := stats.NewExactAUC()
			for _, acc := range accs {
				if err := merged.Merge(acc); err != nil {
					t.Fatal(err)
				}
			}
			auc, err := merged.AUC()
			if err != nil {
				t.Fatal(err)
			}
			curve, err := merged.ROCCurve()
			if err != nil {
				t.Fatal(err)
			}
			pos, neg := merged.TotalWeight()
			snap := make([]metricsSnapshot, len(thresholds))
			for k, thr := range thresholds {
				p, r, fm, tp, fp, merr := merged.MetricsAt(thr)
				if merr != nil {
					t.Fatalf("MetricsAt(%v): %v", thr, merr)
				}
				snap[k] = metricsSnapshot{thr, p, r, fm, tp, fp}
			}

			if first {
				refAUC, refCurve, refPos, refNeg, refMetricsSnap = auc, curve, pos, neg, snap
				first = false
				continue
			}
			if auc != refAUC {
				t.Errorf("shards=%d order=%s: AUC %.17g != %.17g", shardCount, orderName, auc, refAUC)
			}
			if len(curve) != len(refCurve) {
				t.Fatalf("shards=%d order=%s: curve length %d != %d",
					shardCount, orderName, len(curve), len(refCurve))
			}
			for i := range curve {
				if !rocPointEqual(curve[i], refCurve[i]) {
					t.Fatalf("shards=%d order=%s: point %d %+v != %+v",
						shardCount, orderName, i, curve[i], refCurve[i])
				}
			}
			if pos != refPos || neg != refNeg {
				t.Errorf("shards=%d order=%s: totals %v/%v != %v/%v",
					shardCount, orderName, pos, neg, refPos, refNeg)
			}
			// Bit-for-bit equality across all nine configurations.
			for k := range thresholds {
				if !snap[k].equals(refMetricsSnap[k]) {
					t.Errorf("shards=%d order=%s thr=%v: metrics %+v != %+v",
						shardCount, orderName, thresholds[k], snap[k], refMetricsSnap[k])
				}
			}
		}
	}

	// Cross-check the deterministic metrics against an independent plain
	// float64 full-scan. The scan accumulates thousands of additions in
	// arrival order, so its rounding error grows with the sample count
	// (the very drift the exact accumulator removes); a small relative
	// tolerance, independent of n, is the honest comparison.
	const relTol = 1e-12
	for k, thr := range thresholds {
		tp, fp, totalPos := refMetrics(scores, labels, weights, thr)
		got := refMetricsSnap[k]
		if !relClose(got.tp, tp, relTol) || !relClose(got.fp, fp, relTol) {
			t.Errorf("thr=%v: exact tp/fp %v/%v vs plain scan %v/%v",
				thr, got.tp, got.fp, tp, fp)
		}
		wp, wr, wf := classifierFromRef(tp, fp, totalPos)
		if !relCloseNaN(got.precision, wp, relTol) || !relCloseNaN(got.recall, wr, relTol) ||
			!relCloseNaN(got.f1, wf, relTol) {
			t.Errorf("thr=%v: metrics %v/%v/%v vs plain %v/%v/%v",
				thr, got.precision, got.recall, got.f1, wp, wr, wf)
		}
	}

	// The deterministic value must also equal the exact rational ground truth.
	exact, ok := refExactAUC(scores, labels, weights)
	if !ok {
		t.Fatal("reference reports undefined for a two-class dataset")
	}
	if refAUC != exact {
		t.Errorf("streaming AUC %.17g != exact rational %.17g", refAUC, exact)
	}
}

// metricsSnapshot records one MetricsAt result for cross-configuration
// bit-for-bit comparison.
type metricsSnapshot struct {
	threshold float64
	precision float64
	recall    float64
	f1        float64
	tp        float64
	fp        float64
}

func (m metricsSnapshot) equals(o metricsSnapshot) bool {
	return m.threshold == o.threshold && floatEqOrNaN(m.precision, o.precision) &&
		m.recall == o.recall && floatEqOrNaN(m.f1, o.f1) &&
		m.tp == o.tp && m.fp == o.fp
}

// ulpClose reports whether a and b differ by at most ulps units in the last
// place, treating two NaNs as equal.
func ulpClose(a, b float64, ulps int64) bool {
	if a == b || (math.IsNaN(a) && math.IsNaN(b)) {
		return true
	}
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return false
	}
	ia := int64(math.Float64bits(a))
	ib := int64(math.Float64bits(b))
	if ia < 0 {
		ia = -ia
	}
	if ib < 0 {
		ib = -ib
	}
	d := ia - ib
	if d < 0 {
		d = -d
	}
	return d <= ulps
}

// relClose reports whether the relative (or absolute, near zero) difference
// between a and b is within tol.
func relClose(a, b, tol float64) bool {
	if a == b {
		return true
	}
	d := math.Abs(a - b)
	scale := math.Max(math.Abs(a), math.Abs(b))
	if scale == 0 {
		return d <= tol
	}
	return d/scale <= tol
}

// relCloseNaN is relClose with NaN treated as equal to NaN.
func relCloseNaN(a, b, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return relClose(a, b, tol)
}

// classifierFromRef mirrors the accumulator's metric formulas on top of a
// plain full-scan confusion matrix.
func classifierFromRef(tp, fp, totalPos float64) (precision, recall, f1 float64) {
	recall = tp / totalPos
	predicted := tp + fp
	if predicted == 0 {
		return math.NaN(), recall, math.NaN()
	}
	precision = tp / predicted
	if precision+recall == 0 {
		return precision, recall, math.NaN()
	}
	f1 = 2 * precision * recall / (precision + recall)
	return precision, recall, f1
}

func rocPointEqual(a, b stats.ROCPoint) bool {
	return a.Threshold == b.Threshold && a.TPR == b.TPR && a.FPR == b.FPR &&
		floatEqOrNaN(a.Precision, b.Precision) && floatEqOrNaN(a.F1, b.F1) &&
		a.TP == b.TP && a.FP == b.FP
}

func floatEqOrNaN(a, b float64) bool {
	return a == b || (math.IsNaN(a) && math.IsNaN(b))
}

func TestHistogramAUC(t *testing.T) {
	scores := []float64{0.1, 0.2, 0.8, 0.9, 0.5, 0.55}
	labels := []int{0, 0, 1, 1, 1, 0}
	// Axis with 20 bins over [0,1]: every tenth-aligned score lands in its
	// own bin, so the binned AUC must equal the exact AUC with zero bound.
	axis := stats.ScoreAxis{Min: 0, Max: 1, Bins: 20}

	// Axis fine enough to separate every distinct score: binned AUC must
	// equal the exact AUC and the error bound must be zero.
	exact := stats.NewExactAUC()
	if err := exact.AddBatch(stats.Float64Data(scores), labels, nil); err != nil {
		t.Fatal(err)
	}
	exactAUC, err := exact.AUC()
	if err != nil {
		t.Fatal(err)
	}

	h, err := stats.NewHistogramAUC(axis)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.AddBatch(stats.Float64Data(scores), labels, nil); err != nil {
		t.Fatal(err)
	}
	got, err := h.AUC()
	if err != nil {
		t.Fatal(err)
	}
	if got != exactAUC {
		t.Errorf("fine-axis AUC = %.17g, want exact %.17g", got, exactAUC)
	}
	bound, err := h.AUCErrorBound()
	if err != nil || bound != 0 {
		t.Errorf("fine-axis bound = %v, %v; want 0", bound, err)
	}

	// Coarse axis forces mixed bins: the stated bound must cover the gap.
	coarse, err := stats.NewHistogramAUC(stats.ScoreAxis{Min: 0, Max: 1, Bins: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := coarse.AddBatch(stats.Float64Data(scores), labels, nil); err != nil {
		t.Fatal(err)
	}
	cAUC, err := coarse.AUC()
	if err != nil {
		t.Fatal(err)
	}
	cBound, err := coarse.AUCErrorBound()
	if err != nil {
		t.Fatal(err)
	}
	diff := math.Abs(cAUC - exactAUC)
	if diff > cBound*(1+1e-12)+1e-15 {
		t.Errorf("coarse: |binned-exact|=%.3g exceeds bound %.3g", diff, cBound)
	}
	if cBound < 0 || cBound > 0.5 {
		t.Errorf("bound %.3g outside the theoretical [0, 0.5] range", cBound)
	}

	// Invalid axes are rejected at construction.
	for _, bad := range []stats.ScoreAxis{
		{Min: 0, Max: 1, Bins: 0},
		{Min: 1, Max: 1, Bins: 4},
		{Min: math.Inf(1), Max: 1, Bins: 4},
		{Min: math.NaN(), Max: 1, Bins: 4},
	} {
		if _, err := stats.NewHistogramAUC(bad); err != stats.ErrBounds {
			t.Errorf("axis %+v: err = %v, want ErrBounds", bad, err)
		}
	}

	// A score outside the axis is rejected without mutating the state.
	beforePos, beforeNeg := h.TotalWeight()
	if err := h.Add(2, 1, 1); err != stats.ErrBounds {
		t.Errorf("out-of-range Add: err = %v, want ErrBounds", err)
	}
	afterPos, afterNeg := h.TotalWeight()
	if beforePos != afterPos || beforeNeg != afterNeg {
		t.Error("out-of-range Add polluted accumulator state")
	}
}

func TestHistogramAUCMergeAndBound(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	mk := func() *stats.HistogramAUC {
		h, err := stats.NewHistogramAUC(stats.ScoreAxis{Min: 0, Max: 1, Bins: 8})
		if err != nil {
			t.Fatal(err)
		}
		return h
	}
	left, right := mk(), mk()
	all := mk()
	for i := 0; i < 400; i++ {
		s := rng.Float64()
		l := rng.Intn(10) / 9
		w := float64(rng.Intn(50)+1) / 7
		if err := left.Add(s, l, w); err != nil {
			t.Fatal(err)
		}
		if err := all.Add(s, l, w); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 350; i++ {
		s := rng.Float64()
		l := rng.Intn(10) / 9
		w := float64(rng.Intn(50)+1) / 7
		if err := right.Add(s, l, w); err != nil {
			t.Fatal(err)
		}
		if err := all.Add(s, l, w); err != nil {
			t.Fatal(err)
		}
	}

	// Different axis: merge rejected and both sides untouched.
	bad, err := stats.NewHistogramAUC(stats.ScoreAxis{Min: 0, Max: 1, Bins: 7})
	if err != nil {
		t.Fatal(err)
	}
	lp, ln := left.TotalWeight()
	if err := left.Merge(bad); err != stats.ErrAxisMismatch {
		t.Errorf("mismatched merge: err = %v, want ErrAxisMismatch", err)
	}
	if p, n := left.TotalWeight(); p != lp || n != ln {
		t.Error("rejected merge polluted the accumulator")
	}

	if err := left.Merge(right); err != nil {
		t.Fatal(err)
	}
	mergedAUC, err := left.AUC()
	if err != nil {
		t.Fatal(err)
	}
	allAUC, err := all.AUC()
	if err != nil {
		t.Fatal(err)
	}
	if mergedAUC != allAUC {
		t.Errorf("merged AUC %.17g != single-stream %.17g", mergedAUC, allAUC)
	}
	mergedBound, _ := left.AUCErrorBound()
	allBound, _ := all.AUCErrorBound()
	if mergedBound != allBound {
		t.Errorf("merged bound %.17g != single-stream %.17g", mergedBound, allBound)
	}

	// Doubling every sample keeps the per-bin class proportions, so the
	// data-dependent bound must not grow when the sample count doubles.
	rng2 := rand.New(rand.NewSource(11))
	one := mk()
	two := mk()
	draw := func() (float64, int, float64) {
		s := rng2.Float64()
		l := rng2.Intn(10) / 9
		w := float64(rng2.Intn(50)+1) / 7
		return s, l, w
	}
	for i := 0; i < 750; i++ {
		s, l, w := draw()
		if err := one.Add(s, l, w); err != nil {
			t.Fatal(err)
		}
		if err := two.Add(s, l, w); err != nil {
			t.Fatal(err)
		}
	}
	// Second copy of the same multiset doubles the sample count.
	rng2 = rand.New(rand.NewSource(11))
	for i := 0; i < 750; i++ {
		s, l, w := draw()
		if err := two.Add(s, l, w); err != nil {
			t.Fatal(err)
		}
	}
	b1, _ := one.AUCErrorBound()
	b2, _ := two.AUCErrorBound()
	if b2 > b1+1e-15 {
		t.Errorf("bound grew when sample count doubled: %.17g -> %.17g", b1, b2)
	}
}

// TestHistogramErrorBoundIsRigorous exercises the bound across many random
// datasets and bin counts: the exact pairwise AUC must always lie within the
// declared bound, including on coarse axes that force many mixed bins.
func TestHistogramErrorBoundIsRigorous(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for trial := 0; trial < 40; trial++ {
		bins := 1 + rng.Intn(12)
		h, err := stats.NewHistogramAUC(stats.ScoreAxis{Min: 0, Max: 1, Bins: bins})
		if err != nil {
			t.Fatal(err)
		}
		exact := stats.NewExactAUC()
		n := 80 + rng.Intn(200)
		scores := make([]float64, n)
		labels := make([]int, n)
		weights := make([]float64, n)
		for i := 0; i < n; i++ {
			scores[i] = rng.Float64()
			labels[i] = rng.Intn(4) / 3 // 1/4 positives so classes always appear
			weights[i] = float64(rng.Intn(20)+1) / 4
			if err := h.Add(scores[i], labels[i], weights[i]); err != nil {
				t.Fatal(err)
			}
			if err := exact.Add(scores[i], labels[i], weights[i]); err != nil {
				t.Fatal(err)
			}
		}
		binned, err := h.AUC()
		if err != nil {
			t.Fatal(err)
		}
		trueAUC, err := exact.AUC()
		if err != nil {
			t.Fatal(err)
		}
		bound, err := h.AUCErrorBound()
		if err != nil {
			t.Fatal(err)
		}
		gap := math.Abs(binned - trueAUC)
		// The bound is an exact rational rounded to float64 toward nearest,
		// so allow one ULP of slack on the bound itself.
		if gap > bound+math.Abs(bound)*1e-15+1e-16 {
			t.Errorf("trial %d bins %d: gap %.6g exceeds bound %.6g", trial, bins, gap, bound)
		}
		if bound < 0 || bound > 0.5+1e-15 {
			t.Errorf("trial %d: bound %.6g outside [0, 0.5]", trial, bound)
		}
	}
}

func TestHistogramROCCurveAndAxis(t *testing.T) {
	axis := stats.ScoreAxis{Min: 0, Max: 1, Bins: 4}
	h, err := stats.NewHistogramAUC(axis)
	if err != nil {
		t.Fatal(err)
	}
	if h.Axis() != axis {
		t.Errorf("Axis() = %+v, want %+v", h.Axis(), axis)
	}
	for _, c := range []struct {
		score float64
		label int
	}{
		{0.1, 0}, {0.2, 1}, {0.6, 0}, {0.7, 1}, {0.9, 1},
	} {
		if err := h.Add(c.score, c.label, 1); err != nil {
			t.Fatal(err)
		}
	}
	pts, err := h.ROCCurve()
	if err != nil {
		t.Fatal(err)
	}
	first, last := pts[0], pts[len(pts)-1]
	if !math.IsInf(first.Threshold, 1) || first.TPR != 0 || first.FPR != 0 ||
		!math.IsNaN(first.Precision) {
		t.Errorf("first point %+v must start at +Inf with zero rates", first)
	}
	if last.TPR != 1 || last.FPR != 1 {
		t.Errorf("last point %+v must reach (1,1)", last)
	}
	// Thresholds must be descending bin lower edges as the scan walks down.
	for i := 1; i < len(pts); i++ {
		if pts[i].Threshold >= pts[i-1].Threshold {
			t.Errorf("thresholds not descending at %d: %v then %v",
				i, pts[i-1].Threshold, pts[i].Threshold)
		}
	}
	// The trapezoidal area of the returned points matches AUC.
	area := 0.0
	for i := 1; i < len(pts); i++ {
		area += (pts[i].FPR - pts[i-1].FPR) * (pts[i].TPR + pts[i-1].TPR) / 2
	}
	auc, _ := h.AUC()
	if !ulpClose(area, auc, 1) {
		t.Errorf("curve area %.17g != AUC %.17g", area, auc)
	}
}

func TestHistogramUndefinedAndSkipped(t *testing.T) {
	h, err := stats.NewHistogramAUC(stats.ScoreAxis{Min: 0, Max: 1, Bins: 4})
	if err != nil {
		t.Fatal(err)
	}
	// Empty and single-class: undefined, never 0 or 0.5.
	if auc, err := h.AUC(); err != stats.ErrUndefined || !math.IsNaN(auc) {
		t.Errorf("empty AUC = %v, %v; want NaN, ErrUndefined", auc, err)
	}
	if err := h.Add(0.5, 1, 1); err != nil {
		t.Fatal(err)
	}
	if auc, err := h.AUC(); err != stats.ErrUndefined || !math.IsNaN(auc) {
		t.Errorf("single-class AUC = %v, %v; want NaN, ErrUndefined", auc, err)
	}
	// Missing label / non-finite score are counted as skipped, not errors.
	if err := h.Add(0.5, 7, 1); err != nil {
		t.Errorf("missing label: %v", err)
	}
	if err := h.Add(math.NaN(), 0, 1); err != nil {
		t.Errorf("NaN score: %v", err)
	}
	if err := h.Add(math.Inf(1), 0, 1); err != nil {
		t.Errorf("+Inf score: %v", err)
	}
	if h.Skipped() != 3 {
		t.Errorf("Skipped = %d, want 3", h.Skipped())
	}
}

// TestMergeEquivalenceStreaming checks requirement I.7 directly: two
// independently accumulated exact states merged equal both sides fed in
// either order into one accumulator, bit for bit.
func TestMergeEquivalenceStreaming(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	leftS, leftL, leftW := genSamples(rng, 500)
	rightS, rightL, rightW := genSamples(rng, 500)

	l, r := stats.NewExactAUC(), stats.NewExactAUC()
	if err := l.AddBatch(stats.Float64Data(leftS), leftL, stats.Float64Data(leftW)); err != nil {
		t.Fatal(err)
	}
	if err := r.AddBatch(stats.Float64Data(rightS), rightL, stats.Float64Data(rightW)); err != nil {
		t.Fatal(err)
	}
	if err := l.Merge(r); err != nil {
		t.Fatal(err)
	}

	feed := func(reverse bool) *stats.ExactAUC {
		acc := stats.NewExactAUC()
		add := func(s []float64, lab []int, w []float64) {
			idx := indexOrder(len(s), reverse)
			for _, i := range idx {
				if err := acc.Add(s[i], lab[i], w[i]); err != nil {
					t.Fatal(err)
				}
			}
		}
		if reverse {
			add(rightS, rightL, rightW)
			add(leftS, leftL, leftW)
		} else {
			add(leftS, leftL, leftW)
			add(rightS, rightL, rightW)
		}
		return acc
	}
	for _, reverse := range []bool{false, true} {
		other := feed(reverse)
		a1, _ := l.AUC()
		a2, _ := other.AUC()
		if a1 != a2 {
			t.Errorf("reverse=%v AUC %.17g != merged %.17g", reverse, a2, a1)
		}
		p1, _ := l.ROCCurve()
		p2, _ := other.ROCCurve()
		if len(p1) != len(p2) {
			t.Fatalf("curve length mismatch %d != %d", len(p1), len(p2))
		}
		for i := range p1 {
			if !rocPointEqual(p1[i], p2[i]) {
				t.Fatalf("reverse=%v point %d differs: %+v != %+v", reverse, i, p2[i], p1[i])
			}
		}
	}
}

func genSamples(rng *rand.Rand, n int) ([]float64, []int, []float64) {
	s := make([]float64, n)
	l := make([]int, n)
	w := make([]float64, n)
	for i := range s {
		s[i] = math.Floor(rng.Float64()*20) / 4
		l[i] = rng.Intn(3) / 2
		w[i] = float64(rng.Intn(20)+1) / 3
	}
	return s, l, w
}
