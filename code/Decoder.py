import numpy as np
from AModel import AModel
from SGraphModel import SGraphModel
import heapq

class Decoder(object):

    class TrellisSGNode(object):
        def __init__(self, s, p, hmmp, lmp, lm_state, hyp):
            self.s = s
            self.p = p
            self.hmmp = hmmp
            self.lmp = lmp
            self.lm_state = lm_state
            self.hyp = hyp

        def __hash__(self):
            return hash(self.s)   

        def __str__(self):

            return "S: {}, P: {:.5f}".format(self.s, self.p)

    class TrellisHMMNode(object):
        def __init__(self, s, q, p, hmmp, lmp, lm_state, hyp):
            self.s = s
            self.q = q
            self.id = (self.s, self.q)
            self.p = p
            self.hmmp = hmmp
            self.lmp = lmp
            self.hyp = hyp
            self.lm_state = lm_state

        def __hash__(self):
            return hash(self.id)   

        def __str__(self):

            return "S: {}, Q: {}, ID: {}, P: {:.5f}".format(self.s, self.q, self.id, self.p)


    def __init__(self, amodel, sg):

        self.amodel = amodel
        self.sg = sg

        self.actives = []
        self.sg_nodes0 = []
        self.sg_nodes1 = []
        self.sg_null_nodes0 = []
        self.sg_null_nodes1 = []
        self.hmm_nodes0 = []
        self.hmm_nodes1 = []
        self.hmm_actives = {}
        
        self.v_max = -2**10
        self.v_thr = -2**10
        self.v_lm_max = -2**10
        self.v_lm_thr = -2**10
        self.v_abeam= 2**10
        self.v_lm_beam= 2**10
        self.v_maxh = None
        self.hypothesis = []
        self.GSF = 1.0
        self.WIP = 0.0
        self.final_iter = False
        self.hmm_nodes1_heap = []
        self.heap_initialized = False
        self.nmaxstates = 20

        # Enums
        self.NEW = 0
        self.UPDATE = 1


    def expand_sg_nodes(self, node_lst):
        
        print("Expanding nodes:")

        while len(node_lst) != 0:

            node = node_lst.pop()
            sg_state = self.sg.get_state_info(node.s)
            #print(sg_state)

            if sg_state[1] != '-':
                self.hypothesis.append(sg_state[1])
            
            current_p = node.p
            current_lmp = node.lmp
            prev_s = node.s
            edges_begin = sg.edges_begin[prev_s]
            edges_end = sg.edges_end[prev_s]

            for pos in range(edges_begin, edges_end):
                #print("Dst:{}".format(sg.edges_dst[pos]))
                if not self.final_iter:
                    if sg.edges_dst[pos] == sg.final:
                        continue
                
                s = sg.edges_dst[pos]
                p = current_p + sg.edges_weight[pos] * self.GSF + self.WIP
                lmp = current_lmp + sg.edges_weight[pos]
                sgnode = self.TrellisSGNode(s, p, lmp, 0, None, None)
                #print(sgnode)
                self.insert_sg_node(sgnode)
        
        self.sg_null_nodes0 = self.sg_null_nodes1


    def viterbi_init(self, fea):

        #Initialize the search

        sgnode = self.TrellisSGNode(self.sg.start, 0.0, 0.0, 0, None, None)
        self.actives = [-1] * self.sg.nstates
        self.actives[self.sg.start] = 0
        self.sg_null_nodes0 = []
        self.sg_null_nodes0.append(sgnode)

        while len(self.sg_null_nodes0) != 0:
            self.expand_sg_nodes(self.sg_null_nodes0)
            

        self.sg_nodes0 = self.sg_nodes1.copy()

        self.sg_nodes1 = []
        #print("Nodes0 list:")
        #print("\n".join([str(x) for x in self.sg_nodes0]))
        #print("Nodes1 list:")
        #print("\n".join([str(x) for x in self.sg_nodes1]))
        
        # From SG nodes to HMM nodes
        for node in self.sg_nodes0:
            #print(node)
            #print(self.sg.get_state_info(node.s))
            hmmnode = self.TrellisHMMNode(node.s, 0, node.p, node.hmmp, node.lmp, node.lm_state, node.hyp)
            self.insert_hmm_node(hmmnode, fea)

        for node in self.hmm_nodes1:
            if node != None:
                self.hmm_nodes0.append(node)
    
        self.hmm_nodes1 = []

        self.hmm_nodes1_heap = []
        self.heap_initialized = False
        self.hmm_actives = {}

        print("\n".join([str(x) for x in self.hmm_nodes0]))
   
        self.actives = [-1] * self.sg.nstates

        print("Init completed...")
 

    def viterbi(self, fea):

        nodes0 = self.hmm_nodes0
        self.actives = [-1] * self.sg.nstates
        #print("Inside viterbi_iter")
        while len(nodes0) != 0:

            node = nodes0.pop()

            isfinal = False
            #TrellisHMMNode node
            sym = self.sg.get_state_info(node.s)[1]
            auxp = amodel.compute_gmm_emission_prob(fea, sym, node.q)
            #print(auxp)
            node.p+=auxp
            node.hmmp+=auxp

            p1_trans = amodel.get_transitions(sym,node.q)
            # This could be precomputed beforehand when loading the model...
            p0_trans = np.log(1  - np.exp(p1_trans))

            #print(p0_trans)
            #print(p1_trans)
            current_p = node.p
            current_hmmp = node.hmmp

            #Staying in the same state
            hmmnode = self.TrellisHMMNode(node.s, node.q, current_p + p0_trans, current_hmmp + + p0_trans, node.lmp, node.lm_state, node.hyp)
            self.insert_hmm_node(hmmnode, fea)

            if node.q + 1 < amodel.trans_dict[sym].num_hmm_states:
                #Jumping to the following state
                hmmnode = self.TrellisHMMNode(node.s, node.q + 1, current_p + p1_trans, current_hmmp + + p1_trans, node.lmp, node.lm_state, node.hyp)
                self.insert_hmm_node(hmmnode, fea) # from 0->1, 1->2

            else: #from 2-> outside
                isfinal = True

            if isfinal:
                sgnode = self.TrellisSGNode(node.s, node.p, node.hmmp, node.lmp, None, None)
                self.insert_sg_node(sgnode)

        #print([str(x) for x in self.hmm_nodes1])
        
        # At this point, there could be nodes in sg_null_nodes1 and sg_nodes1
        # We should performe sg_null_nodes0 -> sg_nodes0 (viterbi_iter_sg)
        # and sg_nodes0 -> hmm_nodes1 (viterbi_sg2hmm)

        self.sg_null_nodes0 = self.sg_null_nodes1.copy()

        self.sg_null_nodes1 = []
    
        while len(self.sg_null_nodes0) != 0:
            self.expand_sg_nodes(self.sg_null_nodes0)
            #print(self.sg_null_nodes0)

        self.sg_nodes0 = self.sg_nodes1.copy()

        self.sg_nodes1 = []
        #print("Nodes0 list:")
        #print("\n".join([str(x) for x in self.sg_nodes0]))
        #print("Nodes1 list:")
        #print("\n".join([str(x) for x in self.sg_nodes1]))

        # From SG nodes to HMM nodes
        for node in self.sg_nodes0:
            hmmnode = self.TrellisHMMNode(node.s, 0, node.p, node.hmmp, node.lmp, node.lm_state, node.hyp)
            self.insert_hmm_node(hmmnode, fea)

        for node in self.hmm_nodes1:
            if node != None:
                self.hmm_nodes0.append(node)
    
        self.hmm_nodes1 = []

        self.hmm_nodes1_heap = []
        self.heap_initialized = False

        self.hmm_actives = {}

    def heap_push(self, min_heap, node_id, mode):
    
        if mode == self.NEW:
            heapq.heappush(min_heap, node_id)
            return

        if mode == self.UPDATE:
          #Workaround to not implement my own heap...      
          for i,n in enumerate(min_heap):
              if n[1] == node_id[1]:
                  print("Found: {}".format(n))
                  min_heap.pop(i)

          heapq.heappush(min_heap, node_id)
          return 


    def insert_hmm_node(self, hmmnode, fea):
       pass



    def insert_sg_node(self, node):

        actives = self.actives
        sg_state = self.sg.get_state_info(node.s)
        #Is this a word node?
        insert_word = True if sg_state[2] != '-' and sg_state[2] != '>' else False
        
        #Is this a symbol node? 
        if sg_state[1] == '-':
            # Still not symbol node, null_nodes1
            nodes1 = self.sg_null_nodes1
        else:
            # It is a symbol node, nodes1
            nodes1 = self.sg_nodes1

        # If this state was not visited
        if self.actives[node.s] == -1: 

            pos = len(nodes1)
            # Added to actives
            #print("Putting active in pos: {}".format(pos))
            self.actives[node.s] = pos
            # Added to null_nodes1/nodes1 depending on symbol node or not
            nodes1.append(node)
            #print([str(x) for x in nodes1])
            # If this is a word node, restart and manage hypothesis
            if insert_word:
                print(sg_state[2])
                node.hmmp = 0.0
                node.lmp = 0.0
        # Node already inserted, should be updated?
        else:
            #print("IT is active: {}".format(node))
            pos = self.actives[node.s]
            #print([str(x) for x in self.sg_nodes1])
            old_node = self.sg_nodes1[pos]
            #print("Replace?: {}".format(old_node))
            # Update stuff...
            score = old_node.p

            if node.p > score:
                self.sg_nodes1[pos].p = node.p
                #print("Yes")
                #print([str(x) for x in self.sg_nodes1])


    def decode(self, fea):
        
        
        self.viterbi_init(fea.sample[:,0])

        t_max = fea.sample.shape[1]

        #for i in range(1,t_max):
        #for i in range(1,100):

        #    t_fea = fea.sample[:,i]

        #    self.viterbi(t_fea)


if __name__=="__main__":

   import pickle
   amodel = pickle.load( open("../models/monophone_model_I32.p", "rb"))
   sg = pickle.load( open("../models/2.graph.p","rb"))

   from TLSample import TLSample
   sample = TLSample("../samples/AAFA0016.features")
   decoder = Decoder(amodel, sg)

   decoder.decode(sample)
