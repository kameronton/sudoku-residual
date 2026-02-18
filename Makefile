CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra

solver: solver.c
	$(CC) $(CFLAGS) -o solver solver.c

clean:
	rm -f solver

.PHONY: clean
