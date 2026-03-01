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
double gauss(void) {
	double x = (double)rand() / RAND_MAX;
    double y = (double)rand() / RAND_MAX;
    double z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
	return z;
}
int main() {
	srand(1);
	Load();
	init();

	int idx = 0;
	int idx2 = 0;
	for (int ccc = 0; ccc < set[0].NumberTrain; ccc++) {
		int index = 0;
		for (int cc = 0; cc < set[0].Train[ccc].InputHeight; cc++) {
			for (int c = 0; c < set[0].Train[ccc].InputWidth; c++) {
				printf("%%c", set[0].Train[ccc].Input[index] + '0');
				x.X[idx+c] = .1;
				x.X[idx+30+cc] = .1;
				x.X[idx+30+30+set[0].Train[ccc].Input[index]] = .1;
				index++;
				idx += 30+30+10+1;
			}
			for (int c = set[0].Train[ccc].InputWidth; c < 30; c++) {
				printf("f");
				x.X[idx+c] = .1;
				x.X[idx+30+cc] = .1;
				x.X[idx+30+30+10] = .1;
				idx += 30+30+10+1;
			}
			printf("\n");
		}
		for (int cc = set[0].Train[ccc].InputHeight; cc < 30; cc++) {
			for (int c = 0; c < 30; c++) {
				printf("f");
				x.X[idx+c] = .1;
				x.X[idx+30+cc] = .1;
				x.X[idx+30+30+10] = .1;
				idx += 30+30+10+1;
			}
			printf("\n");
		}
		printf("\n");

		index = 0;
		for (int cc = 0; cc < set[0].Train[ccc].OutputHeight; cc++) {
			for (int c = 0; c < set[0].Train[ccc].OutputWidth; c++) {
				printf("%%c", set[0].Train[ccc].Output[index] + '0');
				y.X[idx2+c] = .1;
				y.X[idx2+30+cc] = .1;
				y.X[idx2+30+30+set[0].Train[ccc].Output[index]] = .1;
				index++;
				idx2 += 30+30+10+1;
			}
			for (int c = set[0].Train[ccc].OutputWidth; c < 30; c++) {
				printf("f");
				y.X[idx2+c] = .1;
				y.X[idx2+30+cc] = .1;
				y.X[idx2+30+30+10] = .1;
				idx2 += 30+30+10+1;
			}
			printf("\n");
		}
		for (int cc = set[0].Train[ccc].OutputHeight; cc < 30; cc++) {
			for (int c = 0; c < 30; c++) {
				printf("f");
				y.X[idx2+c] = .1;
				y.X[idx2+30+cc] = .1;
				y.X[idx2+30+30+10] = .1;
				idx2 += 30+30+10+1;
			}
			printf("\n");
		}
		printf("\n");
	}

	for (int ccc = 0; ccc < set[0].NumberTest; ccc++) {
		int index = 0;
		for (int cc = 0; cc < set[0].Test[ccc].InputHeight; cc++) {
			for (int c = 0; c < set[0].Test[ccc].InputWidth; c++) {
				printf("%%c", set[0].Test[ccc].Input[index] + '0');
				x.X[idx+c] = .1;
				x.X[idx+30+cc] = .1;
				x.X[idx+30+30+set[0].Test[ccc].Input[index]] = .1;
				index++;
				idx += 30+30+10+1;
			}
			for (int c = set[0].Test[ccc].InputWidth; c < 30; c++) {
				printf("f");
				x.X[idx+c] = .1;
				x.X[idx+30+cc] = .1;
				x.X[idx+30+30+10] = .1;
				idx += 30+30+10+1;
			}
			printf("\n");
		}
		for (int cc = set[0].Test[ccc].InputHeight; cc < 30; cc++) {
			for (int c = 0; c < 30; c++) {
				printf("f");
				x.X[idx+c] = .1;
				x.X[idx+30+cc] = .1;
				x.X[idx+30+30+10] = .1;
				idx += 30+30+10+1;
			}
			printf("\n");
		}
		printf("\n");
	}
	
	float factor = sqrt(2.0 / ((float)i.W));
	for (int c = 0; c < i.W*i.H; c++) {
		i.X[c] = (float)gauss() * factor * .01;
	}
	factor = sqrt(2.0 / ((float)w0.W));
	for (int c = 0; c < w0.W*w0.H; c++) {
		w0.X[c] = factor * (2 * (float)rand() / ((float)RAND_MAX+1.0) - 1);
	}
	factor = sqrt(2.0 / ((float)w1.W));
	for (int c = 0; c < w1.W*w1.H; c++) {
		w1.X[c] = factor * (2 * (float)rand() / ((float)RAND_MAX+1.0) - 1);
	}
	load();
	for (int i = 0; i < 256; i++) {
		zero();
		gradient();
		adam(i, 1E-3);
	}
	for (int c = 30*30*set[0].NumberTrain; c < y.H; c++) {
		for (int cc = 0; cc < y.W; cc++) {
			printf("%%f ", y.X[c*y.W + cc]);
		}
		printf("\n");
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
	set.Add(&context, "w0", (30 + 30 + 10 + 1), (30 + 30 + 10 + 1))
	set.Add(&context, "b0", (30 + 30 + 10 + 1))
	set.Add(&context, "w1", 2*(30+30+10+1), (30 + 30 + 10 + 1))
	set.Add(&context, "b1", (30 + 30 + 10 + 1))
	set.Add(&context, "i", 7, 30*30*(len(s[0].Train)+len(s[0].Test)))
	set.Add(&context, "x", (30 + 30 + 10 + 1), 30*30*(len(s[0].Train)+len(s[0].Test)))
	set.Add(&context, "y", (30 + 30 + 10 + 1), 30*30*(len(s[0].Train)+len(s[0].Test)))
	set.ByName["x"].Skip = true
	set.ByName["y"].Skip = false
	set.ByName["y"].Set = (30 + 30 + 10 + 1) * 30 * 30 * len(s[0].Train)

	Mul := context.B(context.Mul)
	Add := context.B(context.Add)
	Everett := context.U(context.Everett)
	Quadratic := context.B(context.Quadratic)
	Avg := context.U(context.Avg)
	Dropout := context.U(context.Dropout)
	T := context.U(context.T)
	drop := .3
	dropout := map[string]interface{}{
		"drop": &drop,
	}

	l0 := Everett(Add(Mul(set.Get("w0"), set.Get("x")), set.Get("b0")))
	l1 := Add(Mul(set.Get("w1"), l0), set.Get("b1"))
	out0 := Everett(Add(Mul(set.Get("w0"), set.Get("y")), set.Get("b0")))
	out1 := Add(Mul(set.Get("w1"), out0), set.Get("b1"))
	sa := T(Mul(Dropout(Mul(set.Get("i"), set.Get("i")), dropout), T(l1)))
	loss := Avg(Quadratic(out1, sa))
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
			for r, row := range train.Output {
				for c, value := range row {
					fmt.Fprintf(context.Output, "%d", value)
					if !(c == len(row)-1 && r == len(train.Output)-1) {
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
