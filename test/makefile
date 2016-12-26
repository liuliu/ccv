release: all

debug: all.tests
	${MAKE} -C unit debug ; ${MAKE} -C regression debug ; ${MAKE} -C unit/nnc debug

cover: all.tests
	${MAKE} -C unit cover ; ${MAKE} -C regression cover ; ${MAKE} -C unit/nnc cover

all: all.tests
	${MAKE} -C unit ; ${MAKE} -C regression ; ${MAKE} -C unit/nnc

test:
	${MAKE} -C unit test && ${MAKE} -C regression test && ${MAKE} -C unit/nnc test

check:
	@LLVM_PROFILE_FILE="`pwd`/all.tests.profraw" ./all.tests

clean:
	${MAKE} clean -C ../lib ; ${MAKE} clean -C unit ; ${MAKE} clean -C regression ; ${MAKE} clean -C unit/nnc ; rm -f *.o all.tests

include all.mk
