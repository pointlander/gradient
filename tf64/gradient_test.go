// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tf64

import (
	"math/rand"
	"testing"
)

func TestMul(t *testing.T) {
	a := V{
		X: []float64{1, 2, 3, 4},
		S: []int{2, 2},
		D: make([]float64, 4),
	}
	b := V{
		X: []float64{1, 2},
		S: []int{2, 1},
		D: make([]float64, 2),
	}
	Mul(&a, &b)(func(a *V) {
		if a.X[0] != 5 || a.X[1] != 11 {
			t.Fatal("mul failed", a.X)
		}
	})
	e := V{
		X: []float64{1, 2, 3, 4},
		S: []int{2, 2},
		D: make([]float64, 4),
	}
	Mul(&a, &e)(func(a *V) {
		if a.X[0] != 5 || a.X[1] != 11 || a.X[2] != 11 || a.X[3] != 25 {
			t.Fatal("mul failed", a.X)
		}
	})
}

func TestXORNetwork(t *testing.T) {
	rand.Seed(1)
	random64 := func(a, b float64) float64 {
		return (b-a)*rand.Float64() + a
	}
	input := V{
		X: make([]float64, 2),
		D: make([]float64, 2),
		S: []int{2, 1},
	}
	output := V{
		X: make([]float64, 1),
		D: make([]float64, 1),
		S: []int{1, 1},
	}
	w1 := V{
		X: make([]float64, 4),
		D: make([]float64, 4),
		S: []int{2, 2},
	}
	b1 := V{
		X: make([]float64, 2),
		D: make([]float64, 2),
		S: []int{2, 1},
	}
	w2 := V{
		X: make([]float64, 2),
		D: make([]float64, 2),
		S: []int{2, 1},
	}
	b2 := V{
		X: make([]float64, 1),
		D: make([]float64, 1),
		S: []int{1, 1},
	}
	parameters := []*V{&w1, &b1, &w2, &b2}
	var deltas [][]float64
	for _, p := range parameters {
		for i := range p.X {
			p.X[i] = random64(-1, 1)
		}
		deltas = append(deltas, make([]float64, len(p.X)))
	}
	l1 := SigmoidOp(AddOp(MulOp(w1.Value(), input.Value()), b1.Value()))
	l2 := SigmoidOp(AddOp(MulOp(w2.Value(), l1), b2.Value()))
	d := SubOp(l2, output.Value())
	cost := MulOp(d, d)

	data := [...][3]float64{
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 0},
	}
	alpha, eta := .4, .6
	for i := 0; i < 1000; i++ {
		for i := range data {
			j := i + rand.Intn(len(data)-i)
			data[i], data[j] = data[j], data[i]
		}
		total := 0.0
		for j := range data {
			for _, p := range parameters {
				for k := range p.D {
					p.D[k] = 0
				}
			}
			for k := range input.D {
				input.D[k] = 0
			}
			for k := range output.D {
				output.D[k] = 0
			}
			input.X[0], input.X[1], output.X[0] = data[j][0], data[j][1], data[j][2]
			total += Gradient(cost).X[0]
			for k, p := range parameters {
				for l, d := range p.D {
					deltas[k][l] = alpha*deltas[k][l] - eta*d
					p.X[l] += deltas[k][l]
				}
			}
		}
		t.Log(i, total)
		if total < .001 {
			break
		}
	}
	for i := range data {
		input.X[0], input.X[1] = data[i][0], data[i][1]
		var output V
		l2(func(a *V) {
			output = *a
		})
		if data[i][2] == 1 && output.X[0] < .5 {
			t.Fatal("output should be 1", output.X[0], data[i][0], data[i][1], data[i][2])
		} else if data[i][2] == 0 && output.X[0] >= .5 {
			t.Fatal("output should be 0", output.X[0], data[i][0], data[i][1], data[i][2])
		}
	}
}
