ifneq ($(OBJECTS),)

ifneq ($(PROGRAM),)
build: $(PROGRAM) | $(TGT_DIR)
$(PROGRAM): $(OBJECTS)
endif

DEPENDS := $(patsubst %.o,%.d,$(OBJECTS))
-include $(DEPENDS)

endif

ifneq ($(BINARIES),)

build: $(BINARIES) | $(TGT_DIR)

DEPENDS := $(patsubst %,%.d,$(BINARIES))
-include $(DEPENDS)

endif
