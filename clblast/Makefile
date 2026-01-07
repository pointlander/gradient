all: test_mul/mul test_everett/everett test_avg/avg

test_mul/mul:
	cd test_mul && go build
	cd test_mul && ./test_mul
	gcc -o test_mul/mul test_mul/*.c -lclblast -lOpenCL
	test_mul/mul

test_everett/everett:
	cd test_everett && go build
	cd test_everett && ./test_everett
	gcc -o test_everett/everett test_everett/*.c -lclblast -lOpenCL
	test_everett/everett

test_avg/avg:
	cd test_avg && go build
	cd test_avg && ./test_avg
	gcc -o test_avg/avg test_avg/*.c -lclblast -lOpenCL
	test_avg/avg

clean:
	rm -f test_mul/mul.c test_mul/mul test_mul/test_mul
	rm -f test_everett/everett.c test_everett/everett test_everett/test_everett
	rm -f test_avg/avg.c test_avg/avg test_avg/test_avg
