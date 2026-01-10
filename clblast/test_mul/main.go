// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"github.com/pointlander/gradient/clblast"
	"github.com/pointlander/gradient/tf32"
)

const code = `int main() {
	init();
	for (int i = 0; i < 8; i++) {
		data.X[i] = (float)(i+1);
		data2.X[i] = (float)(i+1);
	}
	gradient();
	for (int i = 0; i < 8; i++) {
		if (data.D[i] != d[i]) {
			printf("%%f != %%f;\n", data.D[i], d[i]);
			exit(1);
		}
		if (data2.D[i] != d2[i]) {
			printf("%%f != %%f;\n", data2.D[i], d2[i]);
			exit(1);
		}
	}
	uninit();
}`

func main() {
	context := clblast.Context{}
	var err error
	context.Output, err = os.Create("mul.c")
	if err != nil {
		panic(err)
	}
	defer context.Output.Close()

	set := clblast.NewSet()
	set.Add(&context, "data", 8, 1)
	set.Add(&context, "data2", 8, 1)

	Everett := context.B(context.Mul)
	loss := Everett(set.Get("data"), set.Get("data2"))
	context.Gradient(set, loss)

	set32 := tf32.NewSet()
	set32.Add("data", 8, 1)
	set32.Add("data2", 8, 1)
	data := set32.ByName["data"]
	data2 := set32.ByName["data2"]
	for i := 0; i < 8; i++ {
		data.X = append(data.X, float32(i+1))
		data2.X = append(data2.X, float32(i+1))
	}
	loss32 := tf32.Mul(set32.Get("data"), set32.Get("data2"))
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
	for _, v := range data.X[:len(data.X)-1] {
		fmt.Fprintf(context.Output, "%f,", v)
	}
	fmt.Fprintf(context.Output, "%f};\n", data.X[len(data.X)-1])
	fmt.Fprintf(context.Output, "float d2[] = {")
	for _, v := range data2.X[:len(data2.X)-1] {
		fmt.Fprintf(context.Output, "%f,", v)
	}
	fmt.Fprintf(context.Output, "%f};\n", data2.X[len(data2.X)-1])
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
