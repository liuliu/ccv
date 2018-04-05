.PHONY: debug undef asan cover

# Debug Scheme

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g -fno-omit-frame-pointer -O0
	LDFLAGS += -g -fno-omit-frame-pointer -O0
endif

debug: CFLAGS += -g -fno-omit-frame-pointer -O0
debug: LDFLAGS += -g -fno-omit-frame-pointer -O0
debug: export DEBUG = 1
debug: all

# Asan Scheme

ASAN ?= 0
ifeq ($(ASAN), 1)
	CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
	LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
endif

asan: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
asan: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
asan: export ASAN = 1
asan: all

# Undefined Scheme

UNDEF ?= 0
ifeq ($(UNDEF), 1)
	CFLAGS += -g -fno-omit-frame-pointer -O0 -fsanitize=address -fsanitize=undefined
	LDFLAGS += -g -fno-omit-frame-pointer -O0 -fsanitize=address -fsanitize=undefined
endif

undef: CFLAGS += -g -fno-omit-frame-pointer -O0 -fsanitize=address -fsanitize=undefined
undef: LDFLAGS += -g -fno-omit-frame-pointer -O0 -fsanitize=address -fsanitize=undefined
undef: export UNDEF = 1
undef: all

# Coverage Scheme

COVER ?= 0
ifeq ($(COVER), 1)
	# -O0 need to be last because the default is -O3
	CFLAGS += -g -fno-omit-frame-pointer -O0 -fprofile-instr-generate -fcoverage-mapping
	LDFLAGS += -g -fno-omit-frame-pointer -O0 -fprofile-instr-generate -fcoverage-mapping
endif

cover: CFLAGS += -g -fno-omit-frame-pointer -O0 -fprofile-instr-generate -fcoverage-mapping
cover: LDFLAGS += -g -fno-omit-frame-pointer -O0 -fprofile-instr-generate -fcoverage-mapping
cover: export COVER = 1
cover: all
