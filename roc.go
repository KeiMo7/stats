package stats

import (
	"math"
	"math/big"
	"sort"
)

// exactFloatSum accumulates a sum of float64 weights EXACTLY, with a memory
// footprint independent of the number of addends.
//
// Every finite float64 x is an exact binary rational. Reading its IEEE 754
// bit pattern yields a 53-bit integer mantissa M (including the implicit
// leading 1 for normal values) and an unbiased exponent e such that
// x == M * 2^e, with e in [-1074, 971]. The accumulator keeps one scale q,
// pinned to the smallest exponent seen, and an integer sum such that the
// sum equals num * 2^q. A plain int64 carries the common case (per-block
// totals of a few hundred to a few thousand weights); the first operation
// that would overflow it promotes the accumulator once to an arbitrary
// precision big.Int, after which it stays exact for arbitrarily large or
// long streams. Integer addition and exact left shifts are associative and
// commutative, so the accumulated value is identical for every arrival
// order and every sharding; the only float64 rounding happens when Float64
// is called. The state size depends on the exponent span and total weight,
// never on the sample count.
type exactFloatSum struct {
	started bool
	small   int64    // valid while big == nil
	q       int      // shared scale exponent
	big     *big.Int // non-nil after promotion out of int64
}

// float64Parts returns the exact integer mantissa M and exponent e of a
// finite, non-zero, non-negative float64 such that x == M * 2^e. Normal
// values carry the implicit leading 1 (M spans 2^52 to 2^53-1); subnormal
// values omit it.
func float64Parts(x float64) (mantissa uint64, exponent int) {
	bits := math.Float64bits(x)
	mantissaBits := bits & (1<<52 - 1)
	expBits := int((bits >> 52) & 0x7ff)
	if expBits == 0 {
		// Subnormal: x = mantissaBits * 2^-1074, no implicit leading bit.
		return mantissaBits, -1074
	}
	// Normal: x = (2^52 | mantissaBits) * 2^(expBits-1023-52).
	return mantissaBits | 1<<52, expBits - 1023 - 52
}

const int64Max = 1<<63 - 1

// promote moves an int64-path accumulator onto the big.Int representation.
func (s *exactFloatSum) promote() {
	s.big = new(big.Int).SetInt64(s.small)
	s.small = 0
}

// addBig folds M * 2^e into the big.Int representation at scale s.q.
func (s *exactFloatSum) addBig(mantissa uint64, e int) {
	term := new(big.Int).SetUint64(mantissa)
	if e < s.q {
		s.big.Lsh(s.big, uint(s.q-e))
		s.q = e
	} else if e > s.q {
		term.Lsh(term, uint(e-s.q))
	}
	s.big.Add(s.big, term)
}

// add folds one finite, non-negative float64 weight into the exact sum.
func (s *exactFloatSum) add(w float64) {
	if w == 0 {
		return
	}
	mantissa, we := float64Parts(w)
	if s.big != nil {
		s.addBig(mantissa, we)
		return
	}
	if !s.started {
		s.small, s.q, s.started = int64(mantissa), we, true
		return
	}
	man := int64(mantissa)
	if we < s.q {
		// Rebase the running integer onto the finer scale.
		shift := s.q - we
		if shift >= 63 || s.small > int64Max>>uint(shift) {
			s.promote()
			s.addBig(mantissa, we)
			return
		}
		s.small <<= uint(shift)
		s.q = we
	} else if we > s.q {
		// Shift only the incoming mantissa onto the current scale.
		shift := we - s.q
		if shift >= 63 || man > int64Max>>uint(shift) {
			s.promote()
			s.addBig(mantissa, we)
			return
		}
		man <<= uint(shift)
	}
	if man > int64Max-s.small { // non-negative addends: overflow
		s.promote()
		s.addBig(mantissa, we)
		return
	}
	s.small += man
}

// merge adds another exact sum without re-rounding through float64.
func (s *exactFloatSum) merge(o *exactFloatSum) {
	if o == nil || !o.started {
		return
	}
	if o.big != nil {
		on := new(big.Int).Set(o.big)
		if s.big == nil {
			if !s.started {
				s.big, s.q, s.started = on, o.q, true
				return
			}
			s.promote()
		}
		if o.q < s.q {
			s.big.Lsh(s.big, uint(s.q-o.q))
			s.q = o.q
		} else if o.q > s.q {
			on.Lsh(on, uint(o.q-s.q))
		}
		s.big.Add(s.big, on)
		return
	}
	if s.big != nil {
		s.addBig(uint64(o.small), o.q)
		return
	}
	if !s.started {
		s.small, s.q, s.started = o.small, o.q, true
		return
	}
	on := o.small
	if o.q < s.q {
		shift := s.q - o.q
		if shift >= 63 || s.small > int64Max>>uint(shift) {
			s.promote()
			s.merge(o)
			return
		}
		s.small <<= uint(shift)
		s.q = o.q
	} else if o.q > s.q {
		shift := o.q - s.q
		if shift >= 63 || on > int64Max>>uint(shift) {
			s.promote()
			s.merge(o)
			return
		}
		on <<= uint(shift)
	}
	if on > int64Max-s.small {
		s.promote()
		s.merge(o)
		return
	}
	s.small += on
}

// Rat returns the exact sum as a rational. The returned value is a copy and
// may be mutated freely by the caller.
func (s *exactFloatSum) Rat() *big.Rat {
	if !s.started {
		return new(big.Rat)
	}
	var num *big.Int
	if s.big != nil {
		num = new(big.Int).Set(s.big)
	} else {
		num = big.NewInt(s.small)
	}
	if s.q >= 0 {
		return new(big.Rat).SetInt(new(big.Int).Lsh(num, uint(s.q)))
	}
	return new(big.Rat).SetFrac(num, new(big.Int).Lsh(big.NewInt(1), uint(-s.q)))
}

// Float64 rounds the exact sum to a single float64. IEEE 754 rounding is
// deterministic, so identical exact sums always round identically.
func (s *exactFloatSum) Float64() float64 {
	f, _ := s.Rat().Float64()
	return f
}

// isZero reports whether nothing has been accumulated.
func (s *exactFloatSum) isZero() bool {
	if !s.started {
		return true
	}
	if s.big != nil {
		return s.big.Sign() == 0
	}
	return s.small == 0
}

// scaledInt returns the exact integer num and scale q with value num*2^q.
// The integer is owned by the accumulator for the small path and freshly
// allocated for the big path; callers must not mutate it.
func (s *exactFloatSum) scaledInt() (*big.Int, int) {
	if s.big != nil {
		return s.big, s.q
	}
	return big.NewInt(s.small), s.q
}

// atScale returns the exact sum as a big.Int expressed at the requested
// (finer or equal) scale q, so block totals at different accumulator scales
// can be combined in one integer domain.
func (s *exactFloatSum) atScale(q int) *big.Int {
	if !s.started {
		return new(big.Int)
	}
	n, ownQ := s.scaledInt()
	if ownQ == q {
		return new(big.Int).Set(n)
	}
	// q is the minimum over all merged blocks, so ownQ >= q always.
	return new(big.Int).Lsh(new(big.Int).Set(n), uint(ownQ-q))
}

// rocBlock holds the exact total positive and negative weight observed at one
// distinct score value (ExactAUC) or inside one histogram bin (HistogramAUC).
type rocBlock struct {
	pos exactFloatSum
	neg exactFloatSum
}

// ROCPoint is one point of a threshold scan.
//
// Threshold is the score boundary at which the point is produced: every
// sample with score >= Threshold is classified positive, so a threshold
// landing inside a block of tied scores takes the whole block (see
// ExactAUC.ROCCurve for the tie rule). TPR is the true positive rate
// (recall), FPR the false positive rate, Precision the positive predictive
// value and F1 the harmonic mean of precision and recall. Precision and F1
// are NaN at a point where no sample is classified positive.
type ROCPoint struct {
	Threshold float64
	TPR       float64
	FPR       float64
	Precision float64
	F1        float64
	TP        float64
	FP        float64
}

// classifierMetrics evaluates precision, recall and their harmonic mean for
// accumulated confusion-matrix weights. Precision and F1 are NaN when no
// weight is classified positive (predictedPos == 0); recall is NaN only when
// no positive weight exists at all, which callers reject before getting here.
func classifierMetrics(tp, fp, totalPos float64) (precision, recall, f1 float64) {
	recall = tp / totalPos
	predictedPos := tp + fp
	if predictedPos == 0 {
		return math.NaN(), recall, math.NaN()
	}
	precision = tp / predictedPos
	if precision+recall == 0 {
		return precision, recall, math.NaN()
	}
	f1 = 2 * precision * recall / (precision + recall)
	return precision, recall, f1
}

// checkWeight validates a single sample weight against the convention shared
// with WeightedMean and PercentileWeighted: negative weights are rejected.
// Zero and missing (NaN) weights are allowed and contribute nothing.
func checkWeight(w float64) error {
	if w < 0 {
		return ErrNegative
	}
	return nil
}

// sortedBlocks returns the score-axis blocks in descending score order, the
// canonical traversal used for every read so results never depend on map
// iteration order or arrival order.
func sortedBlocks(blocks map[float64]rocBlock) ([]float64, []rocBlock) {
	scores := make([]float64, 0, len(blocks))
	for s := range blocks {
		scores = append(scores, s)
	}
	sort.Float64s(scores)
	ordered := make([]rocBlock, len(scores))
	for i, s := range scores {
		ordered[i] = blocks[s]
	}
	// Reverse: highest score first, the direction the threshold scan walks.
	for i, j := 0, len(ordered)-1; i < j; i, j = i+1, j-1 {
		ordered[i], ordered[j] = ordered[j], ordered[i]
		scores[i], scores[j] = scores[j], scores[i]
	}
	return scores, ordered
}

// ExactAUC is a streaming, mergeable accumulator for ROC/AUC statistics over
// weighted binary-classification scores.
//
// State size is linear in the number of DISTINCT score values seen, not in
// the number of samples: every sample carrying the same score is folded into
// one block holding its positive and negative weight. Samples themselves are
// never retained, so the accumulator supports hourly reads over an unbounded
// stream and survives restarts by being merged or serialized by the caller.
//
// Tied scores are handled as indivisible blocks. A threshold that falls in
// the middle of a tie block is resolved by the standard "all or nothing"
// rule: the whole block enters the positive class together, giving the
// trapezoidal ROC AUC with half-credit for tied positive/negative pairs (the
// Mann-Whitney U statistic with tie correction). The rule depends only on
// score ordering, so repeated runs and merges reproduce it identically.
//
// Weight totals are summed exactly as scaled binary integers (see
// exactFloatSum), and every read rounds in one canonical sorted traversal,
// making the output bit-for-bit identical for any sharding or arrival order.
// Scores must be finite; non-finite or missing scores (NaN, +Inf, -Inf) and
// missing labels are skipped, matching the library convention that NaN data
// has no rank.
type ExactAUC struct {
	blocks  map[float64]rocBlock
	skipped int
}

// NewExactAUC creates an empty accumulator.
func NewExactAUC() *ExactAUC {
	return &ExactAUC{blocks: make(map[float64]rocBlock)}
}

// Add folds one labeled, weighted sample into the accumulator.
//
// label is the binary class: 1 for positive, 0 for negative. Any other value
// means the label is missing and the sample is skipped. A NaN or infinite
// score means the score is missing and the sample is skipped. A NaN weight
// is treated as missing (zero contribution); a negative weight is rejected
// with ErrNegative and the accumulator is left untouched.
func (a *ExactAUC) Add(score float64, label int, weight float64) error {
	if err := checkWeight(weight); err != nil {
		return err
	}
	if label != 0 && label != 1 {
		a.skipped++
		return nil
	}
	if math.IsInf(score, 0) || math.IsNaN(score) {
		a.skipped++
		return nil
	}
	if math.IsNaN(weight) {
		weight = 0
	}
	b := a.blocks[score]
	if label == 1 {
		b.pos.add(weight)
	} else {
		b.neg.add(weight)
	}
	a.blocks[score] = b
	return nil
}

// AddBatch folds many samples at once. scores, labels and weights must have
// equal length; weights may be nil, in which case every sample has weight 1.
// On the first invalid sample the batch is rejected as a whole and the
// accumulator is unchanged.
func (a *ExactAUC) AddBatch(scores Float64Data, labels []int, weights Float64Data) error {
	if labels == nil || len(labels) != len(scores) || (weights != nil && len(weights) != len(scores)) {
		return ErrSize
	}
	// Validate before mutating so a rejected batch cannot pollute state.
	for i := range scores {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if err := checkWeight(w); err != nil {
			return err
		}
	}
	for i := range scores {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if err := a.Add(scores[i], labels[i], w); err != nil {
			return err // already validated above; unreachable
		}
	}
	return nil
}

// Skipped reports how many samples were ignored because their label or score
// was missing or non-finite.
func (a *ExactAUC) Skipped() int { return a.skipped }

// Len reports the number of distinct score blocks currently held, which is
// the only dimension the state grows along.
func (a *ExactAUC) Len() int { return len(a.blocks) }

// totals returns the exact positive and negative weight, summing the blocks
// in ascending score order so the rounding order is canonical.
func (a *ExactAUC) totals() (exactFloatSum, exactFloatSum) {
	scores := make([]float64, 0, len(a.blocks))
	for s := range a.blocks {
		scores = append(scores, s)
	}
	sort.Float64s(scores)
	var pos, neg exactFloatSum
	for _, s := range scores {
		b := a.blocks[s]
		pos.merge(&b.pos)
		neg.merge(&b.neg)
	}
	return pos, neg
}

// TotalWeight returns the accumulated positive and negative weight.
func (a *ExactAUC) TotalWeight() (pos, neg float64) {
	p, n := a.totals()
	return p.Float64(), n.Float64()
}

// Merge folds the state of other into a. Both accumulators use exact float64
// score values as their implicit axis, so a merge is always well defined.
// other is not modified. Merging A and B yields exactly the state produced
// by feeding both datasets into one accumulator in any order.
func (a *ExactAUC) Merge(other *ExactAUC) error {
	if other == nil {
		return ErrSize
	}
	for score, ob := range other.blocks {
		b := a.blocks[score]
		b.pos.merge(&ob.pos)
		b.neg.merge(&ob.neg)
		a.blocks[score] = b
	}
	a.skipped += other.skipped
	return nil
}

// exactAUC computes the Mann-Whitney area entirely with arbitrary-precision
// integers: no rational normalization or GCD is performed until the single
// final float64 rounding, which keeps hourly reads cheap while preserving
// the bit-for-bit deterministic value.
//
// At a common scale q (the finest exponent across the blocks), let P_k and
// N_k be the positive and negative weight in score block k. The weighted
// statistic is
//
//	AUC = (sum_k P_k * (N_below_k + N_k/2)) / (P_total * N_total),
//
// where N_below_k is the negative weight in strictly lower-scored blocks.
// Multiplying the numerator by 2 keeps every operand integral.
func exactBlockAUC(blocks map[float64]rocBlock) float64 {
	scores := make([]float64, 0, len(blocks))
	for s := range blocks {
		scores = append(scores, s)
	}
	sort.Float64s(scores) // ascending for the "negatives below" sweep

	// The common scale is the minimum block exponent.
	q, started := 0, false
	consider := func(s *exactFloatSum) {
		if s.started {
			if !started || s.q < q {
				q = s.q
			}
			started = true
		}
	}
	for _, s := range scores {
		b := blocks[s]
		consider(&b.pos)
		consider(&b.neg)
	}
	if !started {
		return math.NaN()
	}

	pTotal, nTotal := new(big.Int), new(big.Int)
	num := new(big.Int) // 2 * (wins + half-credit ties)
	negBelow := new(big.Int)
	tmp := new(big.Int)
	for _, s := range scores {
		b := blocks[s]
		pk := b.pos.atScale(q)
		nk := b.neg.atScale(q)
		pTotal.Add(pTotal, pk)
		nTotal.Add(nTotal, nk)
		// 2*P_k*N_below + P_k*N_k
		tmp.Set(negBelow).Lsh(tmp, 1).Add(tmp, nk).Mul(tmp, pk)
		num.Add(num, tmp)
		negBelow.Add(negBelow, nk)
	}
	den := new(big.Int).Mul(pTotal, nTotal)
	f, _ := new(big.Rat).SetFrac(num, new(big.Int).Lsh(den, 1)).Float64()
	return f
}

// curve walks the sorted blocks once, building the threshold scan. The TPR,
// FPR, precision and F1 of each point are the correctly rounded float64
// values of the exact ratios. AUC() computes the area separately with exact
// integer arithmetic so reads never pay for rational GCDs.
func (a *ExactAUC) curve() []ROCPoint {
	scores, ordered := sortedBlocks(a.blocks)
	posTotal, _ := a.totals()
	totalPos := posTotal.Float64()

	points := make([]ROCPoint, 0, len(ordered)+1)
	// Starting point: nothing classified positive.
	var tp, fp float64
	points = append(points, ROCPoint{
		Threshold: math.Inf(1),
		TPR:       0,
		FPR:       0,
		Precision: math.NaN(),
		F1:        math.NaN(),
	})

	for i, b := range ordered {
		tp += b.pos.Float64()
		fp += b.neg.Float64()
		precision, recall, f1 := classifierMetrics(tp, fp, totalPos)
		points = append(points, ROCPoint{
			Threshold: scores[i],
			TPR:       recall,
			FPR:       fp / a.negTotalFloat(),
			Precision: precision,
			F1:        f1,
			TP:        tp,
			FP:        fp,
		})
	}
	return points
}

// negTotalFloat returns the total negative weight as one rounded float64.
func (a *ExactAUC) negTotalFloat() float64 {
	_, neg := a.totals()
	return neg.Float64()
}

// AUC returns the area under the ROC curve, equal to the weighted
// Mann-Whitney U statistic: the probability that a uniformly chosen positive
// outranks a uniformly chosen negative, with half credit for a tie.
//
// AUC is undefined when either class carries no positive weight (including an
// empty accumulator or a single-class batch); it then returns NaN and
// ErrUndefined rather than 0 or 0.5, because no ranking between the classes
// exists.
func (a *ExactAUC) AUC() (float64, error) {
	pos, neg := a.totals()
	if pos.isZero() || neg.isZero() {
		return math.NaN(), ErrUndefined
	}
	return exactBlockAUC(a.blocks), nil
}

// ROCCurve returns the full threshold scan from (FPR 0, TPR 0) to
// (FPR 1, TPR 1). One point is emitted per distinct score, each point
// classifying every sample at that score and above as positive, so tied
// scores always move together. It returns ErrUndefined when one class is
// absent, for the same reason AUC does.
func (a *ExactAUC) ROCCurve() ([]ROCPoint, error) {
	pos, neg := a.totals()
	if pos.isZero() || neg.isZero() {
		return nil, ErrUndefined
	}
	points := a.curve()
	return points, nil
}

// MetricsAt returns the confusion-matrix weights and the precision, recall
// and F1 obtained by classifying every sample with score >= threshold as
// positive. The whole tie block containing the threshold is taken together
// (equal scores cannot be split), matching the curve. Precision and F1 are
// NaN when nothing is classified positive.
func (a *ExactAUC) MetricsAt(threshold float64) (precision, recall, f1 float64, tp, fp float64, err error) {
	pos, _ := a.totals()
	if pos.isZero() {
		return math.NaN(), math.NaN(), math.NaN(), 0, 0, ErrUndefined
	}
	totalPos := pos.Float64()
	scores, ordered := sortedBlocks(a.blocks)
	var tpe, fpe exactFloatSum
	for i, s := range scores {
		if s >= threshold {
			b := ordered[i]
			tpe.merge(&b.pos)
			fpe.merge(&b.neg)
		}
	}
	tp, fp = tpe.Float64(), fpe.Float64()
	precision, recall, f1 = classifierMetrics(tp, fp, totalPos)
	return precision, recall, f1, tp, fp, nil
}

// ScoreAxis describes the fixed score discretization of a HistogramAUC.
//
// Scores are mapped to Bins equal-width bins over [Min, Max]: bin i covers
// [Min + i*width, Min + (i+1)*width), and the final bin also includes Max.
// Bins must be at least 1 and Min must be strictly less than Max. The axis is
// part of the merge contract: two histogram accumulators can only merge when
// their axes are identical.
type ScoreAxis struct {
	Min  float64
	Max  float64
	Bins int
}

// width returns the bin width, or 0 for an invalid axis.
func (ax ScoreAxis) width() float64 {
	if ax.Bins < 1 || !(ax.Min < ax.Max) || math.IsNaN(ax.Min) || math.IsNaN(ax.Max) ||
		math.IsInf(ax.Min, 0) || math.IsInf(ax.Max, 0) {
		return 0
	}
	return (ax.Max - ax.Min) / float64(ax.Bins)
}

// binOf maps a finite score to its bin index, or -1 when the score falls
// outside the axis.
func (ax ScoreAxis) binOf(score float64) int {
	w := ax.width()
	if w == 0 || score < ax.Min || score > ax.Max {
		return -1
	}
	if score == ax.Max {
		return ax.Bins - 1
	}
	i := int((score - ax.Min) / w)
	if i >= ax.Bins {
		i = ax.Bins - 1
	}
	return i
}

// equals reports whether two axes describe the same discretization.
func (ax ScoreAxis) equals(other ScoreAxis) bool {
	return ax.Min == other.Min && ax.Max == other.Max && ax.Bins == other.Bins
}

// HistogramAUC is a streaming, mergeable ROC/AUC accumulator whose memory is
// bounded by the score-axis resolution rather than the sample count: it
// stores exactly Bins exact positive/negative weight sums no matter how many
// samples arrive. It is the bounded-memory counterpart of ExactAUC for
// continuous score streams where distinct values are effectively unlimited.
//
// Quantizing scores into bins introduces an approximation. AUCErrorBound
// returns a rigorous, data-dependent upper bound on |binned AUC - exact
// AUC| that needs no distributional assumption: every positive/negative pair
// whose two scores land in the same bin is potentially misranked, and each
// such pair contributes at most half (the tie credit) to the error. The
// bound is subadditive under Merge and does not grow when the sample count
// doubles.
type HistogramAUC struct {
	axis    ScoreAxis
	pos     []exactFloatSum
	neg     []exactFloatSum
	skipped int
}

// NewHistogramAUC creates an accumulator on the given axis, returning
// ErrBounds for an invalid axis (fewer than one bin, empty/infinite range).
func NewHistogramAUC(axis ScoreAxis) (*HistogramAUC, error) {
	if axis.width() == 0 {
		return nil, ErrBounds
	}
	return &HistogramAUC{
		axis: axis,
		pos:  make([]exactFloatSum, axis.Bins),
		neg:  make([]exactFloatSum, axis.Bins),
	}, nil
}

// Axis returns the score axis the accumulator was built with.
func (h *HistogramAUC) Axis() ScoreAxis { return h.axis }

// Add folds one sample into its bin. The validation and skipping rules match
// ExactAUC.Add; additionally a finite score outside [Min, Max] is rejected
// with ErrBounds because the axis has no bin for it, and the accumulator is
// left untouched.
func (h *HistogramAUC) Add(score float64, label int, weight float64) error {
	if err := checkWeight(weight); err != nil {
		return err
	}
	if label != 0 && label != 1 {
		h.skipped++
		return nil
	}
	if math.IsInf(score, 0) || math.IsNaN(score) {
		h.skipped++
		return nil
	}
	if math.IsNaN(weight) {
		weight = 0
	}
	bin := h.axis.binOf(score)
	if bin < 0 {
		return ErrBounds
	}
	if label == 1 {
		h.pos[bin].add(weight)
	} else {
		h.neg[bin].add(weight)
	}
	return nil
}

// AddBatch folds many samples at once with the same all-or-nothing semantics
// as ExactAUC.AddBatch: the length and range checks run before any mutation.
func (h *HistogramAUC) AddBatch(scores Float64Data, labels []int, weights Float64Data) error {
	if labels == nil || len(labels) != len(scores) || (weights != nil && len(weights) != len(scores)) {
		return ErrSize
	}
	for i := range scores {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if err := checkWeight(w); err != nil {
			return err
		}
		if (labels[i] == 0 || labels[i] == 1) && !math.IsInf(scores[i], 0) && !math.IsNaN(scores[i]) {
			if h.axis.binOf(scores[i]) < 0 {
				return ErrBounds
			}
		}
	}
	for i := range scores {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if err := h.Add(scores[i], labels[i], w); err != nil {
			return err // validated above; unreachable
		}
	}
	return nil
}

// Skipped reports the count of samples ignored for missing label or score.
func (h *HistogramAUC) Skipped() int { return h.skipped }

// totals sums every bin in ascending bin order, the canonical traversal.
func (h *HistogramAUC) totals() (exactFloatSum, exactFloatSum) {
	var pos, neg exactFloatSum
	for i := range h.pos {
		pos.merge(&h.pos[i])
		neg.merge(&h.neg[i])
	}
	return pos, neg
}

// TotalWeight returns the accumulated positive and negative weight.
func (h *HistogramAUC) TotalWeight() (pos, neg float64) {
	p, n := h.totals()
	return p.Float64(), n.Float64()
}

// Merge folds other into h. The axes must be identical or the merge is
// rejected with ErrAxisMismatch and neither accumulator is modified.
func (h *HistogramAUC) Merge(other *HistogramAUC) error {
	if other == nil {
		return ErrSize
	}
	if !h.axis.equals(other.axis) {
		return ErrAxisMismatch
	}
	for i := range h.pos {
		h.pos[i].merge(&other.pos[i])
		h.neg[i].merge(&other.neg[i])
	}
	h.skipped += other.skipped
	return nil
}

// exactBinAUC is exactBlockAUC over parallel positive/negative weight
// accumulators ordered by ascending score position (distinct blocks or
// histogram bins from the lowest score to the highest).
func exactBinAUC(pos, neg []exactFloatSum) float64 {
	q := 0
	started := false
	consider := func(s *exactFloatSum) {
		if s.started {
			if !started || s.q < q {
				q = s.q
			}
			started = true
		}
	}
	for i := range pos {
		consider(&pos[i])
		consider(&neg[i])
	}

	pTotal, nTotal := new(big.Int), new(big.Int)
	num := new(big.Int)
	negBelow := new(big.Int)
	tmp := new(big.Int)
	for i := range pos {
		pk := pos[i].atScale(q)
		nk := neg[i].atScale(q)
		pTotal.Add(pTotal, pk)
		nTotal.Add(nTotal, nk)
		tmp.Set(negBelow).Lsh(tmp, 1).Add(tmp, nk).Mul(tmp, pk)
		num.Add(num, tmp)
		negBelow.Add(negBelow, nk)
	}
	den := new(big.Int).Mul(pTotal, nTotal)
	f, _ := new(big.Rat).SetFrac(num, new(big.Int).Lsh(den, 1)).Float64()
	return f
}

// curve builds the threshold scan over bins from the highest score down,
// matching ExactAUC.curve. AUC() computes the area separately with exact
// integer arithmetic so reads never pay for rational GCDs.
func (h *HistogramAUC) curve() []ROCPoint {
	posTotal, negTotal := h.totals()
	totalPos := posTotal.Float64()
	totalNeg := negTotal.Float64()

	points := make([]ROCPoint, 0, h.axis.Bins+1)
	var tp, fp float64
	points = append(points, ROCPoint{
		Threshold: math.Inf(1),
		TPR:       0,
		FPR:       0,
		Precision: math.NaN(),
		F1:        math.NaN(),
	})
	w := h.axis.width()
	for i := h.axis.Bins - 1; i >= 0; i-- {
		if h.pos[i].isZero() && h.neg[i].isZero() {
			continue
		}
		tp += h.pos[i].Float64()
		fp += h.neg[i].Float64()
		precision, recall, f1 := classifierMetrics(tp, fp, totalPos)
		// The threshold is the bin's lower edge: score >= edge selects
		// this bin and every bin above it.
		threshold := h.axis.Min + float64(i)*w
		points = append(points, ROCPoint{
			Threshold: threshold,
			TPR:       recall,
			FPR:       fp / totalNeg,
			Precision: precision,
			F1:        f1,
			TP:        tp,
			FP:        fp,
		})
	}
	return points
}

// AUC returns the ROC area computed on the binned score axis. It returns
// ErrUndefined when one class carries no positive weight.
func (h *HistogramAUC) AUC() (float64, error) {
	pos, neg := h.totals()
	if pos.isZero() || neg.isZero() {
		return math.NaN(), ErrUndefined
	}
	return exactBinAUC(h.pos, h.neg), nil
}

// ROCCurve returns the threshold scan over non-empty bins, with the same
// point semantics as ExactAUC.ROCCurve.
func (h *HistogramAUC) ROCCurve() ([]ROCPoint, error) {
	pos, neg := h.totals()
	if pos.isZero() || neg.isZero() {
		return nil, ErrUndefined
	}
	points := h.curve()
	return points, nil
}

// MetricsAt classifies every sample whose bin's lower edge is at or above
// threshold as positive, returning precision, recall, F1 and the confusion
// weights with the same NaN conventions as ExactAUC.MetricsAt.
func (h *HistogramAUC) MetricsAt(threshold float64) (precision, recall, f1 float64, tp, fp float64, err error) {
	pos, _ := h.totals()
	if pos.isZero() {
		return math.NaN(), math.NaN(), math.NaN(), 0, 0, ErrUndefined
	}
	totalPos := pos.Float64()
	w := h.axis.width()
	start := int(math.Ceil((threshold - h.axis.Min) / w))
	if start < 0 {
		start = 0
	}
	if start > h.axis.Bins {
		start = h.axis.Bins
	}
	var tpe, fpe exactFloatSum
	for i := start; i < h.axis.Bins; i++ {
		tpe.merge(&h.pos[i])
		fpe.merge(&h.neg[i])
	}
	tp, fp = tpe.Float64(), fpe.Float64()
	precision, recall, f1 = classifierMetrics(tp, fp, totalPos)
	return precision, recall, f1, tp, fp, nil
}

// AUCErrorBound returns a rigorous upper bound on |binned AUC - exact AUC|.
//
// Quantization can only change the relative order of a positive/negative
// pair whose two scores fall in the same bin; pairs in different bins keep
// their exact ordering. Each such ambiguous pair changes the pairwise
// statistic by at most 0.5 (tie credit). Normalizing by the total pair
// weight gives
//
//	bound = 0.5 * sum_i P_i N_i / ((sum_i P_i)(sum_j N_j)),
//
// where P_i and N_i are the positive and negative weights in bin i. The
// bound needs no assumption about the score distribution, is computed from
// the same O(Bins) state as AUC, is subadditive under merging (unions of
// bins only remove ambiguous pairs), and stays constant or shrinks when the
// sample count doubles with fixed class proportions. It is 0 when no bin
// mixes both classes, in which case the binned AUC is exact.
func (h *HistogramAUC) AUCErrorBound() (float64, error) {
	pos, neg := h.totals()
	if pos.isZero() || neg.isZero() {
		return math.NaN(), ErrUndefined
	}
	// Compute 0.5 * sum_i P_i N_i / (P_total N_total) with exact integers at
	// a common scale so the reported bound is a genuine upper bound rather
	// than one that float64 rounding could push slightly below the truth.
	q, started := 0, false
	consider := func(s *exactFloatSum) {
		if s.started {
			if !started || s.q < q {
				q = s.q
			}
			started = true
		}
	}
	for i := range h.pos {
		consider(&h.pos[i])
		consider(&h.neg[i])
	}
	mixed := new(big.Int)
	pTotal := new(big.Int)
	nTotal := new(big.Int)
	prod := new(big.Int)
	for i := range h.pos {
		pk := h.pos[i].atScale(q)
		nk := h.neg[i].atScale(q)
		prod.Mul(pk, nk)
		mixed.Add(mixed, prod)
		pTotal.Add(pTotal, pk)
		nTotal.Add(nTotal, nk)
	}
	// bound = mixed / (2 * pTotal * nTotal)
	den := new(big.Int).Lsh(new(big.Int).Mul(pTotal, nTotal), 1)
	f, _ := new(big.Rat).SetFrac(mixed, den).Float64()
	return f, nil
}
