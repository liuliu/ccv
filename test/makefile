all:
	${MAKE} -C unit ; ${MAKE} -C regression

test:
	${MAKE} -C unit test && ${MAKE} -C regression test

clean:
	${MAKE} clean -C ../lib ; ${MAKE} clean -C unit ; ${MAKE} clean -C regression
