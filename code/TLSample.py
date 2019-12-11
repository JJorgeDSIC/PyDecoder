import numpy as np

class TLSample(object):


    def __init__(self, sample):

        self.sample = np.zeros((1,1))
        
        with open(sample, "r") as f:
            
            
            line = f.readline()
            
            line = f.readline().split()
            dim = int(line[0])
            length = int(line[1])

            self.sample = np.zeros((dim, length))

            for n in range(length):
                
                self.sample[:,n] = [float(x) for x in f.readline().split()]


if __name__=="__main__":
     
     sample = TLSample("../sample.txt")
