release: all

debug:
	${MAKE} -C unit debug ; ${MAKE} -C regression debug ; ${MAKE} -C unit/nnc debug

undef:
	${MAKE} -C unit undef ; ${MAKE} -C regression undef ; ${MAKE} -C unit/nnc undef

cover:
	${MAKE} -C unit cover ; ${MAKE} -C regression cover ; ${MAKE} -C unit/nnc cover

all:
	${MAKE} -C unit ; ${MAKE} -C regression ; ${MAKE} -C unit/nnc

int:
	${MAKE} -C int/ncc

test:
	${MAKE} -C unit test && ${MAKE} -C regression test && ${MAKE} -C unit/nnc test

check:
	@LLVM_PROFILE_FILE="`pwd`/all.tests.profraw" ./all.tests

clean:
	${MAKE} clean -C ../lib ; ${MAKE} clean -C unit ; ${MAKE} clean -C regression ; ${MAKE} clean -C unit/nnc ; ${MAKE} clean -C int/nnc ; rm -f *.o all.tests

include all.mk
