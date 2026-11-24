package stats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

func TestFloat64Parts(t *testing.T) {
	cases := []float64{
		1, 0.1, 0.2, 0.3, 1.5, 7.1, 6.4,
		math.SmallestNonzeroFloat64, // smallest subnormal
		math.MaxFloat64,
		1e-300,
		3.141592653589793,
	}
	for _, x := range cases {
		man, e := float64Parts(x)
		got := math.Ldexp(float64(man), e)
		if got != x {
			t.Errorf("float64Parts(%g): mantissa %d * 2^%d = %g", x, man, e, got)
		}
	}
}

func TestExactFloatSum(t *testing.T) {
	// Regression test: an earlier Frexp-based representation lost fraction
	// bits, so 0.1+0.2+0.3 silently became 0.4375 instead of 0.6.
	var s exactFloatSum
	for _, w := range []float64{0.1, 0.2, 0.3} {
		s.add(w)
	}
	if got := s.Float64(); got != 0.6 {
		t.Fatalf("0.1+0.2+0.3 = %.17g, want 0.6", got)
	}

	// Exhaustive small-decimal check against an independent big.Rat sum, in
	// randomized order, with weights spanning several binary exponents.
	rng := rand.New(rand.NewSource(3))
	ws := make([]float64, 500)
	want := new(big.Rat)
	for i := range ws {
		ws[i] = float64(rng.Intn(10000)+1) / 1000
		want.Add(want, new(big.Rat).SetFloat64(ws[i]))
	}
	rng.Shuffle(len(ws), func(i, j int) { ws[i], ws[j] = ws[j], ws[i] })

	var acc exactFloatSum
	for _, w := range ws {
		acc.add(w)
	}
	wantF, _ := want.Float64()
	if got := acc.Float64(); got != wantF {
		t.Errorf("random decimal sum = %.17g, want %.17g", got, wantF)
	}

	// Associativity: sharding the same multiset must give the same value.
	var merged exactFloatSum
	for shard := 0; shard < 8; shard++ {
		var part exactFloatSum
		for i := shard; i < len(ws); i += 8 {
			part.add(ws[i])
		}
		merged.merge(&part)
	}
	if merged.Float64() != acc.Float64() {
		t.Errorf("merged %.17g != single stream %.17g", merged.Float64(), acc.Float64())
	}

	// The integer width is driven by the exponent span and total, not the
	// addend count: doubling the addends of one repeated value adds only a
	// handful of bits.
	var rep exactFloatSum
	for i := 0; i < 100000; i++ {
		rep.add(0.125)
	}
	if rep.Float64() != 12500 {
		t.Errorf("repeated sum = %v, want 12500", rep.Float64())
	}
}

func TestExactFloatSumExtremes(t *testing.T) {
	// Values near the float64 range that overflowed an int64-based design
	// must still sum exactly via the arbitrary-precision integer.
	var s exactFloatSum
	huge := math.MaxFloat64 / 4
	const count = 100000
	for i := 0; i < count; i++ {
		s.add(huge)
	}
	want := new(big.Rat).Mul(big.NewRat(count, 1), new(big.Rat).SetFloat64(huge))
	if s.Rat().Cmp(want) != 0 {
		t.Error("extreme-magnitude sum is not exact")
	}
	if got := s.Float64(); !math.IsInf(got, 1) {
		t.Errorf("Float64 of overflowing sum = %v, want +Inf", got)
	}

	// Mixing the smallest subnormals with ordinary weights stays exact and
	// never panics.
	var m exactFloatSum
	m.add(1.0)
	for i := 0; i < 1000; i++ {
		m.add(math.SmallestNonzeroFloat64)
	}
	if m.Float64() != 1.0 {
		t.Errorf("subnormal perturbation changed the rounded sum: %.17g", m.Float64())
	}
	if m.isZero() {
		t.Error("sum reports zero after adding positive weights")
	}

	var empty exactFloatSum
	if !empty.isZero() || empty.Float64() != 0 {
		t.Error("empty accumulator must be zero")
	}
}
