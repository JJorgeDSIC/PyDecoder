import sys
import numpy as np

class SGraphModel(object):
    
    def __init__(self, model_path):
        
        self.sym_to_id = {}
        self.id_to_sym = {}

        self.word_to_id = {}
        self.id_to_word = {}

        self.sym_id = 0
        self.word_id = 0
        self.read_model(model_path)

    def get_state_sym(self, state_id):
        return self.id_to_sym[self.state_sym[state_id]]

    def get_state_word(self, state_id):
        return self.id_to_word[self.state_word[state_id]]

    def get_state_info(self, state_id):
        sym, word, edge_begin, edge_end = self.id_to_sym[self.state_sym[state_id]], \
            self.id_to_word[self.state_word[state_id]], self.edges_begin[state_id], self.edges_end[state_id]
        return (state_id, sym, word, edge_begin, edge_end)
    
    def get_state_info_with_ids(self, state_id):
        sym, word, edge_begin, edge_end = self.state_sym[state_id], \
            self.state_word[state_id], self.edges_begin[state_id], self.edges_end[state_id]
        return (state_id, sym, word, edge_begin, edge_end)

    def read_edges(self,model_file):

        line = model_file.readline().split()
        if len(line) != 3:
            raise Exception("Bad edge format, found: {}".format(" ".join(line)))

        edge_id = int(line[0])
        edge_dst = int(line[1])
        edge_weight = float(line[2])


        self.edges_dst[edge_id] = edge_dst
        self.edges_weight[edge_id] = edge_weight

   
    def read_states(self,model_file):
        
        line = model_file.readline().split()
        if len(line) != 5:
            raise Exception("Wrong format, found: {}".format(" ".join(line)))

        state_id = int(line[0])
        state_sym = line[1].replace("'","")
        state_word= line[2].replace("'","")
        edges_begin = int(line[3])
        edges_end = int(line[4])
        
        if state_sym not in self.sym_to_id:
            self.sym_to_id[state_sym] = self.sym_id
            self.id_to_sym[self.sym_id] = state_sym
            self.sym_id+=1      
        
        sym_id = self.sym_to_id[state_sym]  
      
        if state_word not in self.word_to_id:
            self.word_to_id[state_word] = self.word_id
            self.id_to_word[self.word_id] = state_word
            self.word_id+=1      
        
        word_id = self.word_to_id[state_word]  
      
        self.edges_begin[state_id] = edges_begin        
        self.edges_end[state_id] = edges_end
        self.state_sym[state_id] = sym_id
        self.state_word[state_id] = word_id


    def read_model(self,model_path):

        with open(model_path, "r") as model_file:
            line = model_file.readline()
            if line.strip() != "SG":
                print("KO: SG expected")

            line = model_file.readline().split()
            if line[0] != "NStates":
                print("KO: NStates expected")
            self.nstates = int(line[1])

            line = model_file.readline().split()
            if line[0] != "NEdges":
                print("KO: NEdges expected")
            self.nedges = int(line[1])

            line = model_file.readline().split()
            if line[0] != "Start":
                print("KO: Start expected")
            self.start = int(line[1])

            line = model_file.readline().split()
            if line[0] != "Final":
                print("KO: Final expected")
            self.final= int(line[1])
             
            print("NStates: {}, NEdges {}, Start {}, Final {}".format(
                  self.nstates,
                  self.nedges,
                  self.start,
                  self.final))

            line = model_file.readline().strip()
            if line != "States":
                print("KO: States expected")

            self.edges_begin = np.zeros((self.nstates), dtype=np.int32)
            self.edges_end = np.zeros((self.nstates), dtype=np.int32)
            self.state_sym = np.zeros((self.nstates), dtype=np.int32)
            self.state_word = np.zeros((self.nstates), dtype=np.int32)
            #### READ STATES AND EDGES....
            for n in range(self.nstates):
                self.read_states(model_file)

            print("States read")
           
            line = model_file.readline().strip()
            if line != "Edges":
                print("KO: States expected")

            self.edges_dst = np.zeros((self.nedges), dtype=np.int32)
            self.edges_weight = np.zeros((self.nedges), dtype=np.float32)

            for e in range(self.nedges):
                self.read_edges(model_file)
            
            print("Edge read")

if __name__=="__main__":


    sg = SGraphModel("../models/2.gram.graph")
