all: test_mul/mul test_add/add test_everett/everett test_quadratic/quadratic test_avg/avg test_t/t

test_mul/mul:
	cd test_mul && go build
	cd test_mul && ./test_mul
	nvcc -o test_mul/mul test_mul/*.cu
	test_mul/mul

test_add/add:
	cd test_add && go build
	cd test_add && ./test_add
	nvcc -o test_add/add test_add/*.cu -lOpenCL
	test_add/add

test_everett/everett:
	cd test_everett && go build
	cd test_everett && ./test_everett
	nvcc -o test_everett/everett test_everett/*.cu -lOpenCL
	test_everett/everett

test_quadratic/quadratic:
	cd test_quadratic && go build
	cd test_quadratic && ./test_quadratic
	nvcc -o test_quadratic/quadratic test_quadratic/*.cu -lOpenCL
	test_quadratic/quadratic

test_avg/avg:
	cd test_avg && go build
	cd test_avg && ./test_avg
	nvcc -o test_avg/avg test_avg/*.cu -lOpenCL
	test_avg/avg

test_t/t:
	cd test_t && go build
	cd test_t && ./test_t
	nvcc -o test_t/t test_t/*.cu -lOpenCL
	test_t/t

test_xor/xor:
	cd test_xor && go build
	cd test_xor && ./test_xor
	gcc -g -O1 -fsanitize=address -o test_xor/xor test_xor/*.c -lm -lOpenCL
	test_xor/xor

clean:
	rm -f test_mul/mul.cu test_mul/mul test_mul/test_mul
	rm -f test_add/add.cu test_add/add test_add/test_add
	rm -f test_everett/everett.cu test_everett/everett test_everett/test_everett
	rm -f test_quadratic/quadratic.cu test_quadratic/quadratic test_quadratic/test_quadratic
	rm -f test_avg/avg.cu test_avg/avg test_avg/test_avg
	rm -f test_t/t.cu test_t/t test_t/test_t
	rm -f test_xor/xor.c test_xor/xor test_xor/test_xor
