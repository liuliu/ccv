include config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
CFLAGS := -O3 -ffast-math -Wall -MD $(CFLAGS)
# -fprofile-arcs -ftest-coverage

BUILD_DIR := build/
LIBCCV_PATH := $(BUILD_DIR)lib/libccv.a

.PHONY: all build
all: build
build:

# --- Include directories -----------------------------------------------------

s := bin/
include prefix.mk
include $(s)rules.mk
include suffix.mk

s := lib/
include prefix.mk
include $(s)rules.mk
include suffix.mk

# --- General rules -----------------------------------------------------------

clean:
	rm -rf $(CLEAN)

$(TGT_DIR):
	mkdir -p $(TGT_DIR)

#$(TARGETS): %: %.o ../lib/libccv.a
#	$(CC) -o $@ $< $(LDFLAGS)

$(b)%.o: $(s)%.c
	$(CC) $< -o $@ -c $(CFLAGS)

$(b)%.a:
	ar rcs $@ $^
