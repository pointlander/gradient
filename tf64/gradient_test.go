// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tf64

import (
	"math"
	"math/rand"
	"testing"

	"github.com/pointlander/gradient/sf64"
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

	type Weight struct {
		sf64.V
		Delta float64
	}

	i1, i2, o, weights := sf64.V{}, sf64.V{}, sf64.V{}, [9]Weight{}
	for i := range weights {
		weights[i].X = random64(-1, 1)
	}
	n1 := sf64.SigmoidOp(sf64.AddOp(sf64.AddOp(sf64.MulOp(i1.Value(), weights[0].Value()), sf64.MulOp(i2.Value(), weights[1].Value())), weights[2].Value()))
	n2 := sf64.SigmoidOp(sf64.AddOp(sf64.AddOp(sf64.MulOp(i1.Value(), weights[3].Value()), sf64.MulOp(i2.Value(), weights[4].Value())), weights[5].Value()))
	n3 := sf64.SigmoidOp(sf64.AddOp(sf64.AddOp(sf64.MulOp(n1, weights[6].Value()), sf64.MulOp(n2, weights[7].Value())), weights[8].Value()))
	ds := sf64.SubOp(n3, o.Value())
	costs := sf64.MulOp(ds, ds)

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
	w1.X[0] = weights[0].X
	w1.X[1] = weights[1].X
	b1.X[0] = weights[2].X
	w1.X[2] = weights[3].X
	w1.X[3] = weights[4].X
	b1.X[1] = weights[5].X
	w2.X[0] = weights[6].X
	w2.X[1] = weights[7].X
	b2.X[0] = weights[8].X
	var deltas [][]float64
	for _, p := range parameters {
		deltas = append(deltas, make([]float64, len(p.X)))
	}
	l1 := SigmoidOp(AddOp(MulOp(w1.Value(), input.Value()), b1.Value()))
	l2 := SigmoidOp(AddOp(MulOp(w2.Value(), l1), b2.Value()))
	d := SubOp(l2, output.Value())
	cost := MulOp(d, d)

	round := func(a float64) float64 {
		return math.Round(a*1000000) / 1000000
	}
	compare := func(name string, a, b float64) {
		a, b = round(a), round(b)
		if a != b {
			t.Fatalf("%s %f != %f", name, a, b)
		}
	}

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
		total, totals := 0.0, 0.0
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

			i1.D, i2.D, o.D, i1.X, i2.X, o.X = 0, 0, 0, data[j][0], data[j][1], data[j][2]
			totals += sf64.Gradient(costs).X
			compare("w1 0", w1.D[0], weights[0].D)
			compare("w1 1", w1.D[1], weights[1].D)
			compare("b1 0", b1.D[0], weights[2].D)
			compare("w1 2", w1.D[2], weights[3].D)
			compare("w1 3", w1.D[3], weights[4].D)
			compare("b1 1", b1.D[1], weights[5].D)
			compare("w2 0", w2.D[0], weights[6].D)
			compare("w2 1", w2.D[1], weights[7].D)
			compare("b2 0", b2.D[0], weights[8].D)
			for k := range weights {
				weights[k].Delta, weights[k].D = alpha*weights[k].Delta-eta*weights[k].D, 0
				weights[k].X += weights[k].Delta
			}
		}
		t.Log(i, total, totals)
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
