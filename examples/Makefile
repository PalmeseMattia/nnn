CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

all: and_gate mnist_simple

and_gate: and_gate.c ../nnn.h
	$(CC) $(CFLAGS) and_gate.c -o and_gate $(LDFLAGS)

mnist_simple: mnist_simple.c ../nnn.h
	$(CC) $(CFLAGS) mnist_simple.c -o mnist_simple $(LDFLAGS)

clean:
	rm -f and_gate mnist_simple

.PHONY: all clean
