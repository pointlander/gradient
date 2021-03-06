// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sf64

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestGradient(t *testing.T) {
	var context Context
	v1, v2 := V{0.5, 0}, V{0.4, 0}
	v6 := func(a *V) bool {
		a.D = 1
		return false
	}
	v5 := func(a *V) bool {
		return context.TanH(v6, a)
	}
	v4 := func(a *V) bool {
		return context.Mul(v5, a, &v2)
	}
	context.Add(v4, &v1, &v2)

	w1, w2 := V{0.5, 0}, V{0.4, 0}
	Gradient(TanH(Mul(w2.Meta(), Add(w1.Meta(), w2.Meta()))))

	if fmt.Sprintf("%f", w1.D) != "0.352331" {
		t.Fatalf("w1(%f) != 0.352331", w1.D)
	} else if fmt.Sprintf("%f", w2.D) != "1.145075" {
		t.Fatalf("w1(%f) != 1.145075", w2.D)
	} else if v1.D != w1.D {
		t.Fatalf("v1(%f) != w1(%f)", v1.D, w1.D)
	} else if v2.D != w2.D {
		t.Fatalf("v2(%f) != w2(%f)", v2.D, w1.D)
	}
}

func TestXORNetwork(t *testing.T) {
	rand.Seed(1)
	random64 := func(a, b float64) float64 {
		return (b-a)*rand.Float64() + a
	}
	type Weight struct {
		V
		Delta float64
	}

	i1, i2, o, weights := V{}, V{}, V{}, [9]Weight{}
	for i := range weights {
		weights[i].X = random64(-1, 1)
	}
	n1 := Sigmoid(Add(Add(Mul(i1.Meta(), weights[0].Meta()), Mul(i2.Meta(), weights[1].Meta())), weights[2].Meta()))
	n2 := Sigmoid(Add(Add(Mul(i1.Meta(), weights[3].Meta()), Mul(i2.Meta(), weights[4].Meta())), weights[5].Meta()))
	n3 := Sigmoid(Add(Add(Mul(n1, weights[6].Meta()), Mul(n2, weights[7].Meta())), weights[8].Meta()))
	d := Sub(n3, o.Meta())
	cost := Mul(d, d)

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
			i1.D, i2.D, o.D, i1.X, i2.X, o.X = 0, 0, 0, data[j][0], data[j][1], data[j][2]
			total += Gradient(cost).X
			for k := range weights {
				weights[k].Delta, weights[k].D = alpha*weights[k].Delta-eta*weights[k].D, 0
				weights[k].X += weights[k].Delta
			}
		}
		t.Log(i, total)
		if total < .001 {
			break
		}
	}
	for i := range data {
		i1.X, i2.X = data[i][0], data[i][1]
		var output V
		n3(func(a *V) bool {
			output = *a
			return true
		})
		if data[i][2] == 1 && output.X < .5 {
			t.Log("output should be 1", output.X, data[i][0], data[i][1], data[i][2])
		} else if data[i][2] == 0 && output.X >= .5 {
			t.Log("output should be 0", output.X, data[i][0], data[i][1], data[i][2])
		}
	}
}

func TestSame(t *testing.T) {
	type Dual struct {
		Val, Der float64
	}
	One := Dual{Val: 1.0}

	neg := func(d Dual) Dual {
		return Dual{
			Val: -d.Val,
			Der: -d.Der,
		}
	}
	add := func(u, v Dual) Dual {
		return Dual{
			Val: u.Val + v.Val,
			Der: u.Der + v.Der,
		}
	}
	sub := func(u, v Dual) Dual {
		return Dual{
			Val: u.Val - v.Val,
			Der: u.Der - v.Der,
		}
	}
	mul := func(u, v Dual) Dual {
		return Dual{
			Val: u.Val * v.Val,
			Der: u.Der*v.Val + u.Val*v.Der,
		}
	}
	div := func(u, v Dual) Dual {
		return Dual{
			Val: u.Val / v.Val,
			Der: (u.Der*v.Val - u.Val*v.Der) / (v.Val * v.Val),
		}
	}
	sin := func(d Dual) Dual {
		return Dual{
			Val: math.Sin(d.Val),
			Der: d.Der * math.Cos(d.Val),
		}
	}
	cos := func(d Dual) Dual {
		return Dual{
			Val: math.Cos(d.Val),
			Der: -d.Der * math.Sin(d.Val),
		}
	}
	exp := func(d Dual) Dual {
		return Dual{
			Val: math.Exp(d.Val),
			Der: d.Der * math.Exp(d.Val),
		}
	}
	log := func(d Dual) Dual {
		return Dual{
			Val: math.Log(d.Val),
			Der: d.Der / d.Val,
		}
	}
	sigmoid := func(d Dual) Dual {
		e := exp(d)
		return div(e, add(e, One))
	}
	tanh := func(d Dual) Dual {
		i, j := exp(d), exp(neg(d))
		return div(sub(i, j), add(i, j))
	}

	round := func(a float64) float64 {
		return math.Round(a*1000000) / 1000000
	}
	testA := func(name string, op func(a Meta) Meta, golden func(d Dual) Dual) {
		w1, w2 := V{0.5, 0}, V{0.4, 0}
		Gradient(op(Mul(w2.Meta(), Add(w1.Meta(), w2.Meta()))))

		d1, d2 := Dual{0.5, 0}, Dual{0.4, 0}
		d1.Der = 1
		output := golden(mul(d2, add(d1, d2)))
		if round(output.Der) != round(w1.D) {
			t.Fatalf("a1 %s %f %f", name, output.Der, w1.D)
		}
		d1.Der = 0
		d2.Der = 1
		output = golden(mul(d2, add(d1, d2)))
		if round(output.Der) != round(w2.D) {
			t.Fatalf("a2 %s %f %f", name, output.Der, w2.D)
		}
	}
	testA("sin", Sin, sin)
	testA("cos", Cos, cos)
	testA("exp", Exp, exp)
	testA("log", Log, log)
	testA("sigmoid", Sigmoid, sigmoid)
	testA("tanh", TanH, tanh)
	testB := func(name string, op func(a Meta) Meta, golden func(d Dual) Dual) {
		w1, w2 := V{0.5, 0}, V{0.4, 0}
		Gradient(op(Div(w2.Meta(), Sub(w1.Meta(), w2.Meta()))))

		d1, d2 := Dual{0.5, 0}, Dual{0.4, 0}
		d1.Der = 1
		output := golden(div(d2, sub(d1, d2)))
		if round(output.Der) != round(w1.D) {
			t.Fatalf("b1 %s %f %f", name, output.Der, w1.D)
		}
		d1.Der = 0
		d2.Der = 1
		output = golden(div(d2, sub(d1, d2)))
		if round(output.Der) != round(w2.D) {
			t.Fatalf("b2 %s %f %f", name, output.Der, w2.D)
		}
	}
	testB("sin", Sin, sin)
	testB("cos", Cos, cos)
	testB("exp", Exp, exp)
	testB("log", Log, log)
	testB("sigmoid", Sigmoid, sigmoid)
	testB("tanh", TanH, tanh)
}
