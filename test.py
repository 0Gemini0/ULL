import msgpack

test = [0,1,2,3]
msgpack.dump(test, open("test.txt", 'wb'))
