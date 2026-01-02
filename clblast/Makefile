all: test_avg/avg

test_avg/avg:
	cd test_avg && go build
	cd test_avg && ./test_avg
	gcc -o test_avg/avg test_avg/*.c -lclblast -lOpenCL
	test_avg/avg

clean:
	rm -f test_avg/avg.c test_avg/avg test_avg/test_avg
