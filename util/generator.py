# A generator use case

class Gener(object):
    def __init__(self,num):
        self.data = list(range(num))
        self.cusor = -1
    def __iter__(self):
        print("this is a iter call")
        return self

    def __next__(self):
        self.cusor +=1
        print('next call')
        return self.data[self.cusor]


g =Gener(10)
for i in g:
    print(i)
