// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clblast

import (
	"fmt"
	"os"
)

type (
	// V is a tensor value
	V struct {
		N string // the name
		S []int  // the shape
	}
	// Set is a set of V
	Set struct {
		Weights []*V
		ByName  map[string]*V
	}
	// Continuation is a continuation
	Continuation func(a *V) bool
	// Meta is a function that takes a continuation and return a continuation
	Meta func(k Continuation) Continuation
	// Unary is a unary function
	Unary func(k Continuation, node int, a *V, options ...map[string]interface{}) bool
	// Binary is a binary function
	Binary func(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool
	// Operation is an operation that takes multiple parameters
	Operation func(k Continuation, node int, a ...*V) bool
)

// Panic marks a place we should never get to
func Panic(a *V) bool {
	panic("should not be here")
}

// Meta returns a meta for the value
func (a *V) Meta() Meta {
	return func(k Continuation) Continuation {
		k(a)
		return Panic
	}
}

// NewSet creates a new weight set
func NewSet() Set {
	return Set{
		ByName: make(map[string]*V),
	}
}

// Add adds weights to a set
func (s *Set) Add(context *Context, name string, d ...int) {
	v := context.NewV(d...)
	v.N = name
	s.Weights = append(s.Weights, &v)
	s.ByName[name] = &v
}

// Get gets weights from the set by name
func (s *Set) Get(name string) Meta {
	return s.ByName[name].Meta()
}

// Context is a function context
type Context struct {
	Output *os.File
	Node   int
}

// NewV create a new tensor value
func (context *Context) NewV(s ...int) V {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	return V{
		N: fmt.Sprintf("node%d", context.Node),
		S: s,
	}
}

// Everett computes the split reality activation function
func (context *Context) Everett(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := context.NewV(2*a.S[0], a.S[1])

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	fmt.Fprintf(context.Output, "\tcl_mem device_%s = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
		c.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tcl_mem device_%s_d = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
		c.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, `	cl_int status = clEnqueueFillBuffer(queue, device_%s_d, &pattern_value, pattern_size, 0, %d, 0, NULL, &event);
	if (status == CL_SUCCESS) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
		event = NULL;
	}
`, c.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, `	cl_kernel kernel = clCreateKernel(program, "everett", NULL);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_%s);
	size_t global_work_size[] = {%d};
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &event);
	if (status == CL_SUCCESS) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
		event = NULL;
	}
`, a.N, c.N, a.S[0]*a.S[1])

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	cl_kernel kernel_d = clCreateKernel(program, "everett_d", NULL);
	clSetKernelArg(kernel_d, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel_d, 1, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_d, 2, sizeof(cl_mem), (void*)&device_%s_d);
	size_t global_work_size_a[] = {%d};
	status = clEnqueueNDRangeKernel(queue, kernel_d, 1, NULL, global_work_size_a, NULL, 0, NULL, &event);
	if (status == CL_SUCCESS) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
		event = NULL;
	}
`, c.N, c.N, a.N, c.S[0]*c.S[1])

	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel);\n")
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_d);\n")
	fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s_d);\n", c.N)
	fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s);\n", c.N)

	return false
}

// Avg computes the average of the tensor
func (context *Context) Avg(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := context.NewV(1)

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	fmt.Fprintf(context.Output, "\tcl_mem device_%s = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
		c.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tCLBlastStatusCode status = CLBlastSsum(%d, device_%s, 0, device_%s, 0, 1, &queue, &event);\n",
		c.S[0]*c.S[1], c.N, a.N)
	fmt.Fprintf(context.Output, `	if (status == CLBlastSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
		event = NULL;
	}
`)

	fmt.Fprintf(context.Output, "\tfloat alpha = %ff;\n", 1/float32(c.S[0]*c.S[1]))
	fmt.Fprintf(context.Output, "\tstatus = CLBlastSscal(1, alpha, device_%s, 0, 1, &queue, &event);\n", c.N)
	fmt.Fprintf(context.Output, `	if (status == CLBlastSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
		event = NULL;
	}
`)

	fmt.Fprintf(context.Output, "\tcl_mem device_%s_d = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
		c.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tfloat* host_%s_d = (float*)calloc(1, sizeof(float));\n",
		c.N)
	fmt.Fprintf(context.Output, "clEnqueueWriteBuffer(queue, device_%s_d, CL_TRUE, 0, 1 * sizeof(float), host_%s_d, 0, NULL, NULL);",
		c.N, c.N)

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, "\tstatus = CLBlastSaxpy(1, alpha, device_%s_d, 0, 0, device_%s_d, 0, 1, &queue, &event);\n", c.N, a.N)
	fmt.Fprintf(context.Output, `	if (status == CLBlastSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
		event = NULL;
	}
`)

	fmt.Fprintf(context.Output, "\tfree(host_%s_d);\n", c.N)
	fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s_d);\n", c.N)
	fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s);\n", c.N)

	return false
}

// Op is a operation
func (context *Context) Op(op Operation) func(a ...Meta) Meta {
	return func(a ...Meta) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			var call func(a []Meta, b []*V) (bool, Continuation)
			call = func(a []Meta, b []*V) (bool, Continuation) {
				if len(a) == 0 {
					fmt.Fprintf(context.Output, "\t{\n")
					c := op(k, node, b...)
					fmt.Fprintf(context.Output, "\t}\n")
					return c, nil
				}
				derivatives := false
				continuation := a[0](func(c *V) bool {
					derivatives, _ = call(a[1:], append(b, c))
					return derivatives
				})
				return derivatives, continuation
			}
			_, continuation := call(a, make([]*V, 0, len(a)))
			return continuation
		}
	}
}

// B converts a binary function into an operator
func (context *Context) B(op Binary) func(a, b Meta, options ...map[string]interface{}) Meta {
	return func(a, b Meta, options ...map[string]interface{}) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(a *V) bool {
				derivatives := false
				b(func(b *V) bool {
					fmt.Fprintf(context.Output, "\t{\n")
					derivatives = op(k, node, a, b, options...)
					fmt.Fprintf(context.Output, "\t}\n")
					return derivatives
				})
				return derivatives
			})
		}
	}
}

// U converts a unary function into an operator
func (context *Context) U(op Unary) func(a Meta, options ...map[string]interface{}) Meta {
	return func(a Meta, options ...map[string]interface{}) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(b *V) bool {
				fmt.Fprintf(context.Output, "\t{\n")
				derivatives := op(k, node, b, options...)
				fmt.Fprintf(context.Output, "\t}\n")
				return derivatives
			})
		}
	}
}

// Gradient computes the gradient
func (context *Context) Gradient(set Set, a Meta) (cost V) {
	fmt.Fprintf(context.Output, `//go:build ignore
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
	
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <clblast_c.h>

const char* kernel_source = "__kernel void everett(__global float* input, __global float* output) {\n\
    const int idx = get_global_id(0);\n\
    const int idx2 = 2*idx;\n\
	const float in = input[idx];\n\
	if (in < 0.0f) {\n\
		output[idx2] = 0.0f;\n\
	} else {\n\
		output[idx2] = in;\n\
	}\n\
	if (in > 0.0f) {\n\
		output[idx2+1] = 0.0f;\n\
	} else {\n\
		output[idx2+1] = in;\n\
	}\n\
}\n\
__kernel void everett_d(__global float* cx, __global float* cd, __global float* ad) {\n\
	const int idx = get_global_id(0);\n\
	const int idxA = idx&~1U;\n\
	const int idxB = idx|1U;\n\
	if ((cx[idx] != 0) || ((cx[idxA] == 0) && (cx[idxB] == 0))) {\n\
		ad[idx>>1] += cd[idx];\n\
	}\n\
}\n";

float pattern_value = 0.0f;
size_t pattern_size = sizeof(float);

cl_context context;
cl_command_queue queue;
cl_program program;

struct V {
	int W;
	int H;
	float *X;
	float *D;	
};
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "struct V %s;\n", value.N)
	}

	fmt.Fprintf(context.Output, `void init(void) {
`)
	fmt.Fprintf(context.Output, `	const size_t platform_id = 0;
	const size_t device_id = 0;

	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	printf("%%d\n", num_platforms);
	cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
	clGetPlatformIDs(num_platforms, platforms, NULL);
	cl_platform_id platform = platforms[platform_id];

	cl_uint num_devices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	printf("%%d\n", num_devices);
	cl_device_id* devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	cl_device_id device = devices[device_id];

	context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	queue = clCreateCommandQueue(context, device, 0, NULL);
	size_t kernel_source_length = strlen(kernel_source);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_length, NULL);
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	free(platforms);
	free(devices);	
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\t%s.W = %d;\n", value.N, value.S[0])
		fmt.Fprintf(context.Output, "\t%s.H = %d;\n", value.N, value.S[1])
		fmt.Fprintf(context.Output, "\t%s.X = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\t%s.D = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
	}
	fmt.Fprintf(context.Output, `}
void uninit(void) {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tfree(%s.X);\n", value.N)
		fmt.Fprintf(context.Output, "\tfree(%s.D);\n", value.N)
	}
	fmt.Fprintf(context.Output, `	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
int gradient(void) {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tcl_mem device_%s = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
			value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tclEnqueueWriteBuffer(queue, device_%s, CL_TRUE, 0, %d * sizeof(float), %s.X, 0, NULL, NULL);\n",
			value.N, value.S[0]*value.S[1], value.N)
		fmt.Fprintf(context.Output, "\tcl_mem device_%s_d = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
			value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tclEnqueueWriteBuffer(queue, device_%s_d, CL_TRUE, 0, %d * sizeof(float), %s.D, 0, NULL, NULL);\n",
			value.N, value.S[0]*value.S[1], value.N)
	}
	a(func(a *V) bool {
		fmt.Fprintf(context.Output, "\tfloat* host_%s = (float*)calloc(%d, sizeof(float));\n", a.N, a.S[0]*a.S[1])
		fmt.Fprintf(context.Output, "\tclEnqueueReadBuffer(queue, device_%s, CL_TRUE, 0, %d * sizeof(float), host_%s, 0, NULL, NULL);\n",
			a.N, a.S[0]*a.S[1], a.N)
		fmt.Fprintf(context.Output, "\tfree(host_%s);\n", a.N)
		return false
	})
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s);\n", value.N)
		fmt.Fprintf(context.Output, "\tclEnqueueReadBuffer(queue, device_%s_d, CL_TRUE, 0, %d * sizeof(float), %s.D, 0, NULL, NULL);\n",
			value.N, value.S[0]*value.S[1], value.N)
		fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s_d);\n", value.N)
	}
	fmt.Fprintf(context.Output, `
}
`)
	return
}
