//go:build ignore

#include <stdio.h>

extern void init(void);
extern int gradient(void);
extern void uninit();

int main() {
	init();
	gradient();
	uninit();
}
