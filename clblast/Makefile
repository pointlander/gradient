all: test_everett/everett test_avg/avg

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
	rm -f test_everett/everett.c test_everett/everett test_everett/test_everett
	rm -f test_avg/avg.c test_avg/avg test_avg/test_avg
