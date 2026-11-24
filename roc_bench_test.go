package stats

import (
	"math"
	"math/rand"
	"testing"
)

func benchData() (scores []float64, labels []int, weights []float64) {
	rng := rand.New(rand.NewSource(1))
	const n = 4096
	scores, labels, weights = make([]float64, n), make([]int, n), make([]float64, n)
	for i := range scores {
		scores[i] = math.Floor(rng.Float64()*1000) / 10
		if rng.Intn(100) == 0 {
			labels[i] = 1
		}
		weights[i] = float64(rng.Intn(100)+1) / 10
	}
	return
}

func BenchmarkExactAUCAddOnly(b *testing.B) {
	scores, labels, weights := benchData()
	b.ResetTimer()
	for k := 0; k < b.N; k++ {
		a := NewExactAUC()
		for i := range scores {
			if err := a.Add(scores[i], labels[i], weights[i]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkExactAUCCurveRead(b *testing.B) {
	scores, labels, weights := benchData()
	a := NewExactAUC()
	for i := range scores {
		if err := a.Add(scores[i], labels[i], weights[i]); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for k := 0; k < b.N; k++ {
		if _, err := a.AUC(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkExactAUCAddAndRead covers the full hourly-rollup workload: ingest
// a batch then read the current metric.
func BenchmarkExactAUCAddAndRead(b *testing.B) {
	scores, labels, weights := benchData()
	b.ResetTimer()
	for k := 0; k < b.N; k++ {
		a := NewExactAUC()
		for i := range scores {
			if err := a.Add(scores[i], labels[i], weights[i]); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := a.AUC(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHistogramAUCAddAndRead(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n = 4096
	scores, labels, weights := make([]float64, n), make([]int, n), make([]float64, n)
	for i := range scores {
		scores[i] = rng.Float64()
		if rng.Intn(100) == 0 {
			labels[i] = 1
		}
		weights[i] = float64(rng.Intn(100)+1) / 10
	}
	b.ResetTimer()
	for k := 0; k < b.N; k++ {
		h, err := NewHistogramAUC(ScoreAxis{Min: 0, Max: 1, Bins: 256})
		if err != nil {
			b.Fatal(err)
		}
		for i := range scores {
			if err := h.Add(scores[i], labels[i], weights[i]); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := h.AUC(); err != nil {
			b.Fatal(err)
		}
	}
}
