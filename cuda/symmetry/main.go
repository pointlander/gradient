// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"github.com/pointlander/gradient/cuda"
)

const code = `#include <math.h>
int main() {
	srand(1);
	init();
	float factor = sqrt(2.0 / ((float)i.W));
	/*for (int c = 0; c < i.W*i.H; c++) {
		i.X[c] = factor * (2 * (float)rand() / ((float)RAND_MAX+1.0) - 1);
	}*/
	factor = sqrt(2.0 / ((float)x.W));
	for (int c = 0; c < x.W*x.H; c++) {
		x.X[c] = factor * (2 * (float)rand() / ((float)RAND_MAX+1.0) - 1);
	}
	load();
	for (int i = 0; i < 128; i++) {
		zero();
		gradient();
		adam(i, 1E-3);
	}
	store();
	uninit();
}`

func main() {
	context := cuda.Context{}
	var err error
	context.Output, err = os.Create("sym.cu")
	if err != nil {
		panic(err)
	}
	defer context.Output.Close()

	set := cuda.NewSet()
	set.Add(&context, "input", 2, 4)
	set.Add(&context, "output", 1, 4)
	set.ByName["input"].Skip = true
	set.ByName["output"].Skip = true
	set.Add(&context, "i", 7, 4000)
	set.Add(&context, "x", 32, 4000)

	Mul := context.B(context.Mul)
	//Add := context.B(context.Add)
	//Everett := context.U(context.Everett)
	Quadratic := context.B(context.Quadratic)
	Avg := context.U(context.Avg)
	Dropout := context.U(context.Dropout)
	T := context.U(context.T)
	drop := .3
	dropout := map[string]interface{}{
		"drop": &drop,
	}

	sa := T(Mul(Dropout(Mul(set.Get("i"), set.Get("i")), dropout), T(set.Get("x"))))
	loss := Avg(Quadratic(set.Get("x"), sa))
	context.Gradient(set, loss)

	fmt.Fprintf(context.Output, `void callback(float* output, int w, int h) {
	printf("%%f\n", output[0]);
}
`)
	fmt.Fprintf(context.Output, code)
}
