.PHONY: debug 

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
