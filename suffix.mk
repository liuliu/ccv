ifneq ($(OBJECTS),)

ifneq ($(PROGRAM),)
build: $(PROGRAM) | $(TGT_DIR)
$(PROGRAM): $(OBJECTS)
endif

$(b)%.o: $(s)%.c | $(TGT_DIR); $(CC) $(CFLAGS) -c -o $@ $<

DEPENDS := $(patsubst %.o,%.d,$(OBJECTS))
-include $(DEPENDS)

endif

ifneq ($(BINARIES),)

build: $(BINARIES) | $(TGT_DIR)

$(b)%.o: $(s)%.c | $(TGT_DIR); $(CC) $(CFLAGS) -c -o $@ $<

DEPENDS := $(patsubst %,%.d,$(BINARIES))
-include $(DEPENDS)

endif
