all: test_mul/mul test_add/add test_everett/everett test_quadratic/quadratic test_avg/avg test_xor/xor

test_mul/mul:
	cd test_mul && go build
	cd test_mul && ./test_mul
	gcc -o test_mul/mul test_mul/*.c -lclblast -lOpenCL
	test_mul/mul

test_add/add:
	cd test_add && go build
	cd test_add && ./test_add
	gcc -g -O1 -fsanitize=address -o test_add/add test_add/*.c -lclblast -lOpenCL
	test_add/add

test_everett/everett:
	cd test_everett && go build
	cd test_everett && ./test_everett
	gcc -o test_everett/everett test_everett/*.c -lclblast -lOpenCL
	test_everett/everett

test_quadratic/quadratic:
	cd test_quadratic && go build
	cd test_quadratic && ./test_quadratic
	gcc -g -O1 -fsanitize=address -o test_quadratic/quadratic test_quadratic/*.c -lclblast -lOpenCL
	test_quadratic/quadratic

test_avg/avg:
	cd test_avg && go build
	cd test_avg && ./test_avg
	gcc -o test_avg/avg test_avg/*.c -lclblast -lOpenCL
	test_avg/avg

test_xor/xor:
	cd test_xor && go build
	cd test_xor && ./test_xor
	gcc -g -O1 -fsanitize=address -o test_xor/xor test_xor/*.c -lm -lclblast -lOpenCL
	test_xor/xor

clean:
	rm -f test_mul/mul.c test_mul/mul test_mul/test_mul
	rm -f test_add/add.c test_add/add test_add/test_add
	rm -f test_everett/everett.c test_everett/everett test_everett/test_everett
	rm -f test_quadratic/quadratic.c test_quadratic/quadratic test_quadratic/test_quadratic
	rm -f test_avg/avg.c test_avg/avg test_avg/test_avg
	rm -f test_xor/xor.c test_xor/xor test_xor/test_xor
