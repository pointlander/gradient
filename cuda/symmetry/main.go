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

const data = `
struct Example {
	int InputWidth;
	int InputHeight;
	char* Input;
	int OutputWidth;
	int OutputHeight;
	char* Output;
};
struct Set {
	int NumberTest;
	struct Example* Test;
	int NumberTrain;
	struct Example* Train;
};
struct Set* set;
void Load() {
`

const code = `#include <math.h>
int main() {
	srand(1);
	Load();
	init();
	float factor = sqrt(2.0 / ((float)i.W));
	int index = 0;
	for (int cc = 0; cc < set[0].Train[0].InputHeight; cc++) {
		for (int c = 0; c < set[0].Train[0].InputWidth; c++) {
			printf("%%c", set[0].Train[0].Input[index] + '0');
			index++;
		}
		printf("\n");
	}
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
	fmt.Fprintf(context.Output, data)
	fmt.Fprintf(context.Output, "\tset = (Set*)calloc(%d, sizeof(struct Set));\n", len(s))
	index := 0
	for i, v := range s {
		fmt.Fprintf(context.Output, "\tset[%d].NumberTest = %d;\n", i, len(v.Test))
		fmt.Fprintf(context.Output, "\tset[%d].Test = (struct Example*)calloc(%d, sizeof(struct Example));\n", i, len(v.Test))
		for ii, test := range v.Test {
			width := len(test.Input[0])
			height := len(test.Input)
			fmt.Fprintf(context.Output, "\tset[%d].Test[%d].InputWidth = %d;\n", i, ii, width)
			fmt.Fprintf(context.Output, "\tset[%d].Test[%d].InputHeight = %d;\n", i, ii, height)
			fmt.Fprintf(context.Output, "\tstatic char a_%d[] = {", index)
			for r, row := range test.Input {
				for c, value := range row {
					fmt.Fprintf(context.Output, "%d", value)
					if !(c == len(row)-1 && r == len(test.Input)-1) {
						fmt.Fprintf(context.Output, ",")
					}
				}
			}
			fmt.Fprintf(context.Output, "};\n")
			fmt.Fprintf(context.Output, "\tset[%d].Test[%d].Input = a_%d;\n", i, ii, index)
			index++
			width = len(test.Output[0])
			height = len(test.Output)
			fmt.Fprintf(context.Output, "\tset[%d].Test[%d].OutputWidth = %d;\n", i, ii, width)
			fmt.Fprintf(context.Output, "\tset[%d].Test[%d].OutputHeight = %d;\n", i, ii, height)
			fmt.Fprintf(context.Output, "\tstatic char a_%d[] = {", index)
			for r, row := range test.Input {
				for c, value := range row {
					fmt.Fprintf(context.Output, "%d", value)
					if !(c == len(row)-1 && r == len(test.Input)-1) {
						fmt.Fprintf(context.Output, ",")
					}
				}
			}
			fmt.Fprintf(context.Output, "};\n")
			fmt.Fprintf(context.Output, "\tset[%d].Test[%d].Output = a_%d;\n", i, ii, index)
			index++
		}
		fmt.Fprintf(context.Output, "\tset[%d].NumberTrain = %d;\n", i, len(v.Train))
		fmt.Fprintf(context.Output, "\tset[%d].Train = (struct Example*)calloc(%d, sizeof(struct Example));\n", i, len(v.Train))
		for ii, train := range v.Train {
			width := len(train.Input[0])
			height := len(train.Input)
			fmt.Fprintf(context.Output, "\tset[%d].Train[%d].InputWidth = %d;\n", i, ii, width)
			fmt.Fprintf(context.Output, "\tset[%d].Train[%d].InputHeight = %d;\n", i, ii, height)
			fmt.Fprintf(context.Output, "\tstatic char a_%d[] = {", index)
			for r, row := range train.Input {
				for c, value := range row {
					fmt.Fprintf(context.Output, "%d", value)
					if !(c == len(row)-1 && r == len(train.Input)-1) {
						fmt.Fprintf(context.Output, ",")
					}
				}
			}
			fmt.Fprintf(context.Output, "};\n")
			fmt.Fprintf(context.Output, "\tset[%d].Train[%d].Input = a_%d;\n", i, ii, index)
			index++
			width = len(train.Output[0])
			height = len(train.Output)
			fmt.Fprintf(context.Output, "\tset[%d].Train[%d].OutputWidth = %d;\n", i, ii, width)
			fmt.Fprintf(context.Output, "\tset[%d].Train[%d].OutputHeight = %d;\n", i, ii, height)
			fmt.Fprintf(context.Output, "\tstatic char a_%d[] = {", index)
			for r, row := range train.Input {
				for c, value := range row {
					fmt.Fprintf(context.Output, "%d", value)
					if !(c == len(row)-1 && r == len(train.Input)-1) {
						fmt.Fprintf(context.Output, ",")
					}
				}
			}
			fmt.Fprintf(context.Output, "};\n")
			fmt.Fprintf(context.Output, "\tset[%d].Train[%d].Output = a_%d;\n", i, ii, index)
			index++
		}
	}
	fmt.Fprintf(context.Output, "}\n")
	fmt.Fprintf(context.Output, code)
}
