all:
	make -C lib && make -C bin

clean:
	make -C lib clean && make -C bin clean

