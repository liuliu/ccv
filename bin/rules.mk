BINARIES := \
	$(b)bbfcreate \
	$(b)bbfdetect \
	$(b)bbffmt \
	$(b)convert \
	$(b)dpmcreate \
	$(b)dpmdetect \
	$(b)msermatch \
	$(b)siftmatch \
	$(b)swtcreate \
	$(b)swtdetect \
	$(b)tld \

$(BINARIES): $(LIBCCV_PATH)

$(BINARIES): LDFLAGS += -L$(BUILD_DIR)lib -lccv
$(BINARIES): CFLAGS += -Ilib

.PRECIOUS: $(patsubst %,%.o,$(BINARIES))
