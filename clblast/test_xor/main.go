// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"github.com/pointlander/gradient/clblast"
)

const code = `#include <math.h>
int main() {
	init();
	input.X[0] = 0;
	input.X[1] = 0;
	input.X[2] = 1;
	input.X[3] = 0;
	input.X[4] = 0;
	input.X[5] = 1;
	input.X[6] = 1;
	input.X[7] = 1;
	output.X[0] = 0;
	output.X[1] = 1;
	output.X[2] = 1;
	output.X[3] = 0;
	float factor = sqrt(2.0 / ((float)w0.W));
	for (int i = 0; i < w0.W*w0.H; i++) {
		w0.X[i] = factor * (2 * (float)rand() / ((float)RAND_MAX+1.0) - 1);
	}
	for (int i = 0; i < b0.W*b0.H; i++) {
		b0.X[i] = 0.0;
	}
	factor = sqrt(2.0 / ((float)w1.W));
	for (int i = 0; i < w1.W*w1.H; i++) {
		w1.X[i] = factor * (2 * (float)rand() / ((float)RAND_MAX+1.0) - 1);
	}
	for (int i = 0; i < b1.W*b1.H; i++) {
		b1.X[i] = 0.0;
	}
	for (int i = 0; i < 256; i++) {
		for (int i = 0; i < w0.W*w0.H; i++) {
			w0.D[i] = 0;
		}
		for (int i = 0; i < b0.W*b0.H; i++) {
			b0.D[i] = 0;
		}
		for (int i = 0; i < w1.W*w1.H; i++) {
			w1.D[i] = 0;
		}
		for (int i = 0; i < b1.W*b1.H; i++) {
			b1.D[i] = 0;
		}
		gradient();
		for (int i = 0; i < w0.W*w0.H; i++) {
			w0.X[i] -= .05*w0.D[i];
		}
		for (int i = 0; i < b0.W*b0.H; i++) {
			b0.X[i] -= .05*b0.D[i];
		}
		for (int i = 0; i < w1.W*w1.H; i++) {
			w1.X[i] -= .05*w1.D[i];
		}
		for (int i = 0; i < b1.W*b1.H; i++) {
			b1.X[i] -= .05*b1.D[i];
		}
	}
	uninit();
}`

func main() {
	context := clblast.Context{}
	var err error
	context.Output, err = os.Create("xor.c")
	if err != nil {
		panic(err)
	}
	defer context.Output.Close()

	set := clblast.NewSet()
	set.Add(&context, "input", 2, 4)
	set.Add(&context, "output", 1, 4)
	set.Add(&context, "w0", 2, 4)
	set.Add(&context, "b0", 4)
	set.Add(&context, "w1", 8, 1)
	set.Add(&context, "b1", 1)

	Mul := context.B(context.Mul)
	Add := context.B(context.Add)
	Everett := context.U(context.Everett)
	Quadratic := context.B(context.Quadratic)
	Avg := context.U(context.Avg)
	l1 := Everett(Add(Mul(set.Get("w0"), set.Get("input")), set.Get("b0")))
	l2 := Add(Mul(set.Get("w1"), l1), set.Get("b1"))
	loss := Avg(Quadratic(l2, set.Get("output")))
	context.Gradient(set, loss)

	fmt.Fprintf(context.Output, `void callback(float* output, int w, int h) {
	printf("%%f\n", output[0]);
}
`)
	fmt.Fprintf(context.Output, code)
}
