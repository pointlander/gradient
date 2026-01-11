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
		if (i&1) { 
			data.X[i] = -1;
		} else {
			data.X[i] = 1;
		}
	}
	gradient();
	for (int i = 0; i < 8; i++) {
		if (data.D[i] != d[i]) {
			printf("d %%f != %%f;\n", data.D[i], d[i]);
			exit(1);
		}
	}
	uninit();
}`

func main() {
	context := clblast.Context{}
	var err error
	context.Output, err = os.Create("everett.c")
	if err != nil {
		panic(err)
	}
	defer context.Output.Close()

	set := clblast.NewSet()
	set.Add(&context, "data", 8, 1)

	Everett := context.U(context.Everett)
	loss := Everett(set.Get("data"))
	context.Gradient(set, loss)

	set32 := tf32.NewSet()
	set32.Add("data", 8, 1)
	data := set32.ByName["data"]
	for i := 0; i < data.S[0]*data.S[1]; i++ {
		if i&1 == 1 {
			data.X = append(data.X, -1)
		} else {
			data.X = append(data.X, 1)
		}
	}
	loss32 := tf32.Everett(set32.Get("data"))
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
