# Compiler configuration
CC = clang
CFLAGS = -Wall -Wextra -O3 -std=c99

# Target binary name
TARGET = tensor_test

# Source and object files
SRCS = tensor.c main.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Compile C source files into object files
%.o: %.c tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Run the test binary instantly
run: all
	./$(TARGET)

.PHONY: all clean run


