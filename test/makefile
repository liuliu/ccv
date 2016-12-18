release: all

debug:
	${MAKE} -C unit debug ; ${MAKE} -C regression debug ; ${MAKE} -C unit/nnc debug

cover:
	${MAKE} -C unit cover ; ${MAKE} -C regression cover ; ${MAKE} -C unit/nnc cover

all:
	${MAKE} -C unit ; ${MAKE} -C regression ; ${MAKE} -C unit/nnc

test:
	${MAKE} -C unit test && ${MAKE} -C regression test && ${MAKE} -C unit/nnc test

clean:
	${MAKE} clean -C ../lib ; ${MAKE} clean -C unit ; ${MAKE} clean -C regression
