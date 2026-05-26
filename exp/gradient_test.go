// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestMul(t *testing.T) {
	a := NewV[float64](2, 2)
	a.Set([]float64{1, 2, 3, 4})
	b := NewV[float64](2)
	b.Set([]float64{1, 2})
	var context Context[float64]
	context.Clear()
	context.Mul(func(a *V[float64]) bool {
		if a.X[0] != 5 || a.X[1] != 11 {
			t.Fatal("mul failed", a.X)
		}
		return false
	}, 0, a, b)
	e := NewV[float64](2, 2)
	e.Set([]float64{1, 2, 3, 4})
	context.Mul(func(a *V[float64]) bool {
		if a.X[0] != 5 || a.X[1] != 11 || a.X[2] != 11 || a.X[3] != 25 {
			t.Fatal("mul failed", a.X)
		}
		return true
	}, 1, a, e)
}

func TestXORNetwork(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	random64 := func(a, b float64) float64 {
		return (b-a)*rng.Float64() + a
	}

	type Weight struct {
		W     *V[float64]
		Delta float64
	}

	ctxt := Context[float64]{}
	XSigmoid := ctxt.U(ctxt.Sigmoid)
	XAdd := ctxt.B(ctxt.Add)
	XSub := ctxt.B(ctxt.Sub)
	XMul := ctxt.B(ctxt.Mul)

	i1, i2, o, weights := NewV[float64](1), NewV[float64](1), NewV[float64](1), [9]Weight{}
	i1.X = i1.X[:cap(i1.X)]
	i2.X = i2.X[:cap(i2.X)]
	o.X = o.X[:cap(o.X)]
	for i := range weights {
		weights[i].W = NewV[float64](1)
		weights[i].W.X = weights[i].W.X[:cap(weights[i].W.X)]
		weights[i].W.X[0] = random64(-1, 1)
	}
	n1 := XSigmoid(XAdd(XAdd(XMul(i1.Meta(), weights[0].W.Meta()), XMul(i2.Meta(), weights[1].W.Meta())), weights[2].W.Meta()))
	n2 := XSigmoid(XAdd(XAdd(XMul(i1.Meta(), weights[3].W.Meta()), XMul(i2.Meta(), weights[4].W.Meta())), weights[5].W.Meta()))
	n3 := XSigmoid(XAdd(XAdd(XMul(n1, weights[6].W.Meta()), XMul(n2, weights[7].W.Meta())), weights[8].W.Meta()))
	ds := XSub(n3, o.Meta())
	half := V[float64]{S: []int{1, 1}, X: []float64{.5}, D: make([]float64, 1)}
	costs := XMul(XMul(ds, ds), half.Meta())

	context := Context[float64]{}
	input, output := NewV[float64](2), NewV[float64](1)
	w1, b1 := NewV[float64](2, 2), NewV[float64](2)
	w2, b2 := NewV[float64](2), NewV[float64](1)
	parameters := []*V[float64]{w1, b1, w2, b2}
	w1.Set([]float64{weights[0].W.X[0], weights[1].W.X[0], weights[3].W.X[0], weights[4].W.X[0]})
	b1.Set([]float64{weights[2].W.X[0], weights[5].W.X[0]})
	w2.Set([]float64{weights[6].W.X[0], weights[7].W.X[0]})
	b2.Set([]float64{weights[8].W.X[0]})
	var deltas [][]float64
	for _, p := range parameters {
		deltas = append(deltas, make([]float64, len(p.X)))
	}
	Sigmoid := context.U(context.Sigmoid)
	Add := context.B(context.Add)
	Mul := context.B(context.Mul)
	Quadratic := context.B(context.Quadratic)
	l1 := Sigmoid(Add(Mul(w1.Meta(), input.Meta()), b1.Meta()))
	l2 := Sigmoid(Add(Mul(w2.Meta(), l1), b2.Meta()))
	cost := Quadratic(l2, output.Meta())

	round := func(a float64) float64 {
		return math.Round(a*1e6) / 1e6
	}
	compare := func(name string, a, b float64) {
		a, b = round(a), round(b)
		if a != b {
			//t.Fatalf("%s %f != %f", name, a, b)
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
				p.Zero()
			}
			input.Set(data[j][:2])
			output.Set(data[j][2:])
			context.Clear()
			total += Gradient(cost).X[0]
			for k, p := range parameters {
				for l, d := range p.D {
					deltas[k][l] = alpha*deltas[k][l] - eta*d
					p.X[l] += deltas[k][l]
				}
			}

			i1.D[0], i2.D[0], o.D[0], i1.X[0], i2.X[0], o.X[0] = 0, 0, 0, data[j][0], data[j][1], data[j][2]
			totals += Gradient(costs).X[0]
			compare("w1 0", w1.D[0], weights[0].W.D[0])
			compare("w1 1", w1.D[1], weights[1].W.D[0])
			compare("b1 0", b1.D[0], weights[2].W.D[0])
			compare("w1 2", w1.D[2], weights[3].W.D[0])
			compare("w1 3", w1.D[3], weights[4].W.D[0])
			compare("b1 1", b1.D[1], weights[5].W.D[0])
			compare("w2 0", w2.D[0], weights[6].W.D[0])
			compare("w2 1", w2.D[1], weights[7].W.D[0])
			compare("b2 0", b2.D[0], weights[8].W.D[0])
			for k := range weights {
				weights[k].Delta, weights[k].W.D[0] = alpha*weights[k].Delta-eta*weights[k].W.D[0], 0
				weights[k].W.X[0] += weights[k].Delta
			}
		}
		t.Log(i, total, totals)
		if total < .001 {
			break
		}
	}
	for i := range data {
		input.X[0], input.X[1] = data[i][0], data[i][1]
		var output V[float64]
		context.Clear()
		l2(func(a *V[float64]) bool {
			output = *a
			return true
		})
		if data[i][2] == 1 && output.X[0] < .5 {
			t.Fatal("output should be 1", output.X[0], data[i][0], data[i][1], data[i][2])
		} else if data[i][2] == 0 && output.X[0] >= .5 {
			t.Fatal("output should be 0", output.X[0], data[i][0], data[i][1], data[i][2])
		}
	}
}

func TestXORNetworkFull(t *testing.T) {
	data := [...][3]float64{
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 0},
	}
	context := Context[float64]{}
	set := context.NewSet()
	set.Add("w0", 2, 8)
	set.AddBias("b0", 8)
	set.Add("w1", 16, 1)
	set.AddBias("b1", 1)
	set.AddData("input", 2, 4)
	set.AddData("output", 1, 4)
	rng := rand.New(rand.NewSource(2))
	set.InitAdam(rng)
	input := set.ByName["input"]
	output := set.ByName["output"]
	index0, index1 := 0, 0
	for _, d := range data {
		input.X[index0] = d[0]
		index0++
		input.X[index0] = d[1]
		index0++
		output.X[index1] = d[2]
		index1++
	}
	Add := context.B(context.Add)
	Mul := context.B(context.Mul)
	Everett := context.U(context.Everett)
	Sigmoid := context.U(context.Sigmoid)
	Quadratic := context.B(context.Quadratic)
	Avg := context.U(context.Avg)
	l0 := Everett(Add(Mul(set.Get("w0"), set.Get("input")), set.Get("b0")))
	l1 := Sigmoid(Add(Mul(set.Get("w1"), l0), set.Get("b1")))
	loss := Avg(Quadratic(l1, set.Get("output")))
	for range 33 {
		set.Zero()
		l := Gradient(loss)
		set.Adam(B1, B2, .7)
		t.Log(l)
	}

	l1(func(a *V[float64]) bool {
		x := a.X
		t.Log(x)
		for i, d := range data {
			if d[2] == 0 && x[i] > .5 {
				t.Fatal("incorrect", 0, x[i])
			} else if d[2] == 1 && x[i] <= .5 {
				t.Fatal("incorrect", 1, x[i])
			}
		}
		return true
	})
}
