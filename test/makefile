all:
	${MAKE} -C functional ; ${MAKE} -C regression

test:
	${MAKE} -C functional test && ${MAKE} -C regression test

clean:
	${MAKE} clean -C ../lib ; ${MAKE} clean -C functional ; ${MAKE} clean -C regression
