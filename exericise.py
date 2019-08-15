import random
import numpy as np
train_x = [([np.array([1, 2, 3]), np.array([5, 6, 7])], [0, 1], [1, 2]), ([np.array([5, 6, 7]), np.array([7, 9, 10])], [2, 1], [5, 6]),
           ([np.array([6, 7, 8]), np.array([9, 10, 11])], [9, 10], [17, 19])]
train_y = [0, 1, 2]

randnum = random.randint(0,2)
random.seed(randnum)
random.shuffle(train_x)


random.seed(randnum)
random.shuffle(train_y)

print("train_x:{}".format(train_x))
print("train_y:{}".format(train_y))
for (con, p1, p2), l in zip(train_x, train_y):
    print("con:{}".format(con))
    print("p1:{}".format(p1))
    print("p2:{}".format(p2))
    print("l:{}".format(l))
