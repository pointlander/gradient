// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/parser"
	"go/printer"
	"go/token"
	"io/ioutil"
	"math"
	"os"
	"text/template"
)

// Gradient is a gradient template
type Gradient struct {
	input   string // the file containing the template
	output  string // the output file
	Package string // the name of the package
	Type    string // the base type
}

// Execute runs the template
func (g *Gradient) Execute() {
	input, err := ioutil.ReadFile(g.input)
	if err != nil {
		panic(err)
	}
	tmpl, err := template.New(g.Package).Parse(string(input))
	if err != nil {
		panic(err)
	}
	buffer := bytes.Buffer{}
	err = tmpl.Execute(&buffer, g)
	if err != nil {
		panic(err)
	}

	output, err := os.Create(g.output)
	if err != nil {
		panic(err)
	}
	defer output.Close()

	fileSet := token.NewFileSet()
	code, err := parser.ParseFile(fileSet, g.output, &buffer, parser.ParseComments)
	if err != nil {
		buffer.WriteTo(output)
		panic(fmt.Errorf("%v: %v", g.output, err))
	}

	formatter := printer.Config{Mode: printer.TabIndent | printer.UseSpaces, Tabwidth: 8}
	err = formatter.Fprint(output, fileSet, code)
	if err != nil {
		buffer.WriteTo(output)
		panic(fmt.Errorf("%v: %v", g.output, err))
	}
}

var (
	// LFSR find LFSR polynomials
	LFSR = flag.Bool("lfsr", false, "find LFSR polynomials")
)

func main() {
	flag.Parse()

	if *LFSR {
		// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
		// https://users.ece.cmu.edu/~koopman/lfsr/index.html
		count, polynomial := 0, uint32(0x80000000)
		for polynomial != 0 {
			lfsr, period := uint32(1), 0
			for {
				lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & polynomial)
				period++
				if lfsr == 1 {
					break
				}
			}
			fmt.Printf("%v period=%v\n", count, period)
			if period == math.MaxUint32 {
				fmt.Printf("%x\n", polynomial)
				return
			}
			count++
			polynomial++
		}
		return
	}

	gradients := []Gradient{
		{
			input:   "float_gradient.t",
			output:  "float/gradient.go",
			Package: "float",
			Type:    "big.Float",
		},
		{
			input:   "scalar_gradient.t",
			output:  "sf64/gradient.go",
			Package: "sf64",
			Type:    "float64",
		},
		{
			input:   "scalar_gradient.t",
			output:  "sf32/gradient.go",
			Package: "sf32",
			Type:    "float32",
		},
		{
			input:   "scalar_gradient.t",
			output:  "sc128/gradient.go",
			Package: "sc128",
			Type:    "complex128",
		},
		{
			input:   "tensor_gradient.t",
			output:  "tf64/gradient.go",
			Package: "tf64",
			Type:    "float64",
		},
		{
			input:   "tensor_gradient.t",
			output:  "tf32/gradient.go",
			Package: "tf32",
			Type:    "float32",
		},
		{
			input:   "tensor_gradient.t",
			output:  "tc128/gradient.go",
			Package: "tc128",
			Type:    "complex128",
		},
	}
	for _, gradient := range gradients {
		gradient.Execute()
	}
}
