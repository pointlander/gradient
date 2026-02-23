// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
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

// Example is a learning example
type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

// Set is a set of examples
type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

// Load loads the data
func Load() []Set {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make([]Set, len(dirs))
	for i, dir := range dirs {
		data, err := os.ReadFile("ARC-AGI/data/training/" + dir.Name())
		if err != nil {
			panic(err)
		}
		err = json.Unmarshal(data, &sets[i])
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("loaded", len(sets))
	test, train := 0, 0
	for _, set := range sets {
		test += len(set.Test)
		train += len(set.Train)
	}
	fmt.Println("test", test)
	fmt.Println("train", train)
	return sets
}

func main() {
	s := Load()
	_ = s
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
