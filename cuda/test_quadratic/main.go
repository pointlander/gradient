// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"github.com/pointlander/gradient/cuda"
	"github.com/pointlander/gradient/tf32"
)

const code = `int main() {
	init();
	for (int i = 0; i < 4; i++) {
		data.X[i] = (float)(i+1);
	}
	for (int i = 0; i < 4; i++) {
		data2.X[i] = (float)(i+1);
	}
	load();
	gradient();
	store();
	for (int i = 0; i < 4; i++) {
		if (data.D[i] != d[i]) {
			printf("d %%f != %%f;\n", data.D[i], d[i]);
			exit(1);
		}
	}
	for (int i = 0; i < 4; i++) {
		if (data2.D[i] != d2[i]) {
			printf("d2 %%f != %%f;\n", data2.D[i], d2[i]);
			exit(1);
		}
	}
	uninit();
}`

func main() {
	context := cuda.Context{}
	var err error
	context.Output, err = os.Create("quadratic.cu")
	if err != nil {
		panic(err)
	}
	defer context.Output.Close()

	set := cuda.NewSet()
	set.Add(&context, "data", 2, 2)
	set.Add(&context, "data2", 2, 2)

	Quadratic := context.B(context.Quadratic)
	loss := Quadratic(set.Get("data"), set.Get("data2"))
	context.Gradient(set, loss)

	set32 := tf32.NewSet()
	set32.Add("data", 2, 2)
	set32.Add("data2", 2, 2)
	data := set32.ByName["data"]
	data2 := set32.ByName["data2"]
	for i := 0; i < data.S[0]*data.S[1]; i++ {
		data.X = append(data.X, float32(i+1))
	}
	for i := 0; i < data2.S[0]*data2.S[1]; i++ {
		data2.X = append(data2.X, float32(i+1))
	}
	loss32 := tf32.Quadratic(set32.Get("data"), set32.Get("data2"))
	loss32(func(a *tf32.V) bool {
		fmt.Fprintf(context.Output, "float x[] = {")
		for _, v := range a.X[:len(a.X)-1] {
			fmt.Fprintf(context.Output, "%f,", v)
		}
		fmt.Fprintf(context.Output, "%f};\n", a.X[len(a.X)-1])
		for i := range a.D {
			a.D[i] = 1
		}
		return false
	})
	fmt.Fprintf(context.Output, "float d[] = {")
	for _, v := range data.D[:len(data.D)-1] {
		fmt.Fprintf(context.Output, "%f,", v)
	}
	fmt.Fprintf(context.Output, "%f};\n", data.D[len(data.D)-1])
	fmt.Fprintf(context.Output, "float d2[] = {")
	for _, v := range data2.D[:len(data2.D)-1] {
		fmt.Fprintf(context.Output, "%f,", v)
	}
	fmt.Fprintf(context.Output, "%f};\n", data2.D[len(data2.D)-1])
	fmt.Fprintf(context.Output, `void callback(float* output, int w, int h) {
	for (int i = 0; i < w*h; i++) {
		if (x[i] != output[i]) {
			printf("%%f != %%f;\n", x[i], output[i]);
			exit(1);
		}
	}
}
`)
	fmt.Fprintf(context.Output, code)
}
