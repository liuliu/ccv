include config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
CFLAGS := -O3 -ffast-math -Wall -MD $(CFLAGS)
#LDFLAGS := -Wl,-O1 -Wl,--as-needed $(LDFLAGS)
# -fprofile-arcs -ftest-coverage

BUILD_DIR := build/
LIBCCV_PATH := $(BUILD_DIR)lib/libccv.so

# --- Remove unused builtin rules ---------------------------------------------
%.c: %.w %.ch
%:: RCS/%,v
%:: RCS/%
%:: SCCS/s.%
%:: %,v
%:: s.%
MAKEFLAGS += -Rr

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

s := test/
include prefix.mk
include $(s)rules.mk
include suffix.mk

# --- General rules -----------------------------------------------------------

clean:
	rm -rf $(CLEAN)

$(TGT_DIR):
	mkdir -p $(TGT_DIR)

$(BUILD_DIR)%.a:
	ar rcs $@ $^

$(BUILD_DIR)%.so: $(BUILD_DIR)%.o | $(TGT_DIR)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)%.o: %.c | $(TGT_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)%: $(BUILD_DIR)%.o | $(TGT_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
