class base:

    def __init__(self, a, b):

        self.a = a
        self.b = b

    def __str__(self):
        print("a = {}, b = {}".format(a,b))


class A(base):

    def __str__(self):

        print("a = {}".format(self.a))

if __name__ == "__main__":

    x = A(1,2)
    print(x)
