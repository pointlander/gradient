// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"github.com/pointlander/gradient/clblast"
)

const code = `int main() {
	init();
	gradient();
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
	fmt.Fprintf(context.Output, `void callback(float* output, int w, int h) {
}
`)
	fmt.Fprintf(context.Output, code)
}
