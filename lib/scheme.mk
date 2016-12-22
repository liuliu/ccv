.PHONY: debug cover

# Debug Scheme

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined
	LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined
endif

debug: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined
debug: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined
debug: export DEBUG = 1
debug: all


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
