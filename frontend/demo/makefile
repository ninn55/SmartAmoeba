CC = gcc
INCLUDES = -I$(SRC_DIRS)
CFLAGS = -std=c99 -O2 -g $(INCLUDES)
# CFLAGS = -std=c99 -O2 -Wall -g $(INCLUDES)

MAINSRC = main.c
TARGET_EXEC = main

BUILD_DIR = ./build
SRC_DIRS = .

# SRCS = $(shell find $(SRC_DIRS) -name *.c)
SRCS = $(wildcard $(SRC_DIRS)/*.c)
SRCS := $(filter-out $(SRC_DIRS)/$(MAINSRC), $(SRCS))
OBJS = $(SRCS:%=$(BUILD_DIR)/%.o)
LIBS = -lm

.PHONY: 
	clean

all: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.c.o: %.c $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET_EXEC): $(MAINSRC) $(OBJS) $(BUILD_DIR)
	$(CC) -o $@ $< $(OBJS) $(CFLAGS)  $(LIBS)

clean:
	rm -rf $(BUILD_DIR)/*