TEST_TARGETS := \
	$(b)functional/3rdparty.tests \
	$(b)functional/algebra.tests \
	$(b)functional/basic.tests \
	$(b)functional/io.tests \
	$(b)functional/memory.tests \
	$(b)functional/numeric.tests \
	$(b)functional/transform.tests \
	$(b)functional/util.tests \
	$(b)regression/defects.l0.1.tests \

TGT_DIR += \
	$(b)functional \
	$(b)regression \

$(TEST_TARGETS): $(LIBCCV_PATH)

$(TEST_TARGETS): LDFLAGS += -L$(BUILD_DIR)lib -lccv
$(TEST_TARGETS): CFLAGS += -Ilib

build: $(TEST_TARGETS)

test: build
	for test in $(TEST_TARGETS); do ./"$Ftest"; done
