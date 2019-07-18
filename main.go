// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/parser"
	"go/printer"
	"go/token"
	"io/ioutil"
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

func main() {
	gradients := []Gradient{
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
