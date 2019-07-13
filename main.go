// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

type NumR struct {
	x float64
	d float64
}

func Panic(a *NumR) {
	panic("should not be here")
}

func (a *NumR) Value() Meta {
	return func(k Continuation) Continuation {
		k(a)
		return Panic
	}
}

type Continuation func(a *NumR)
type Unary func(a *NumR) func(k Continuation)
type Binary func(a, b *NumR) func(k Continuation)
type Meta func(k Continuation) Continuation

func Add(a, b *NumR) func(k Continuation) {
	return func(k Continuation) {
		c := NumR{a.x + b.x, 0}
		k(&c)
		a.d += c.d
		b.d += c.d
	}
}

func Mul(a, b *NumR) func(k Continuation) {
	return func(k Continuation) {
		c := NumR{a.x * b.x, 0}
		k(&c)
		a.d += b.x * c.d
		b.d += a.x * c.d
	}
}

func TanH(a *NumR) func(k Continuation) {
	return func(k Continuation) {
		i, j := math.Exp(a.x), math.Exp(-a.x)
		c := NumR{(i - j) / (i + j), 0}
		k(&c)
		a.d += (1 - c.x*c.x) * c.d
	}
}

func basic() {
	v1, v2 := NumR{0.5, 0}, NumR{0.4, 0}
	v6 := func(a *NumR) {
		a.d = 1
	}
	v5 := func(a *NumR) {
		TanH(a)(v6)
	}
	v4 := func(a *NumR) {
		Mul(a, &v2)(v5)
	}
	v3 := Add(&v1, &v2)
	v3(v4)
	fmt.Println(v1)
	fmt.Println(v2)
}

func B(op Binary) func(a, b Meta) Meta {
	return func(a, b Meta) Meta {
		return func(k Continuation) Continuation {
			return b(func(b *NumR) {
				a(func(a *NumR) {
					op(a, b)(k)
				})
			})
		}
	}
}

func U(op Unary) func(a Meta) Meta {
	return func(a Meta) Meta {
		return func(k Continuation) Continuation {
			return a(func(b *NumR) {
				op(b)(k)
			})
		}
	}
}

var (
	AddOp  = B(Add)
	MulOp  = B(Mul)
	TanHOp = U(TanH)
)

func Grad(a Meta) Continuation {
	return a(func(a *NumR) {
		a.d = 1
	})
}

func advanced() {
	v1, v2 := NumR{0.5, 0}, NumR{0.4, 0}
	Grad(TanHOp(MulOp(v2.Value(), AddOp(v1.Value(), v2.Value()))))
	fmt.Println(v1)
	fmt.Println(v2)
}

func main() {
	basic()
	advanced()
}
