import numpy as np
from models.AModel import AModel
from models.SGraphModel import SGraphModel
import heapq
import logging

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
        self.nmaxhyp = 20

        # Enums
        self.NEW = 0
        self.UPDATE = 1


    def expand_sg_nodes(self, node_lst):
        
        logging.debug("Expanding nodes:")

        while len(node_lst) != 0:

            node = node_lst.pop()
            sg_state = self.sg.get_state_info(node.s)
            #logging.debug(sg_state)

            if sg_state[1] != '-':
                self.hypothesis.append(sg_state[1])
            
            current_p = node.p
            current_lmp = node.lmp
            prev_s = node.s
            edges_begin = sg.edges_begin[prev_s]
            edges_end = sg.edges_end[prev_s]

            for pos in range(edges_begin, edges_end):
                #logging.debug("Dst:{}".format(sg.edges_dst[pos]))
                if not self.final_iter:
                    if sg.edges_dst[pos] == sg.final:
                        continue
                
                s = sg.edges_dst[pos]
                p = current_p + sg.edges_weight[pos] * self.GSF + self.WIP
                lmp = current_lmp + sg.edges_weight[pos]
                sgnode = self.TrellisSGNode(s, p, lmp, 0, None, None)
                #logging.debug(sgnode)
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
        #logging.debug("Nodes0 list:")
        #logging.debug("\n".join([str(x) for x in self.sg_nodes0]))
        #logging.debug("Nodes1 list:")
        #logging.debug("\n".join([str(x) for x in self.sg_nodes1]))
        
        # From SG nodes to HMM nodes
        for node in self.sg_nodes0:
            #logging.debug(node)
            #logging.debug(self.sg.get_state_info(node.s))
            hmmnode = self.TrellisHMMNode(node.s, 0, node.p, node.hmmp, node.lmp, node.lm_state, node.hyp)
            self.insert_hmm_node(hmmnode, fea)

        for node in self.hmm_nodes1:
            if node != None:
                self.hmm_nodes0.append(node)
    
        self.hmm_nodes1 = []

        self.hmm_nodes1_heap = []
        self.heap_initialized = False
        self.hmm_actives = {}

        #logging.debug("\n".join([str(x) for x in self.hmm_nodes0]))
   
        self.actives = [-1] * self.sg.nstates

        logging.debug("Init completed...")
 

    def viterbi(self, fea):

        # hmm_nodes0 <- hmm_nodes1
        self.nodes0 = []
        # From the worst to the best node, reversed
        while(len(self.hmm_nodes1_heap) != 0):
            self.nodes0.append(heapq.heappop(self.hmm_nodes1_heap))

        nodes0 = self.hmm_nodes0
        self.hmm_nodes1 = []
        self.hmm_actives = {}
        #self.hmm_nodes1_heap should be empty 
        assert(len(self.hmm_nodes1_heap) == 0)


        self.actives = [-1] * self.sg.nstates
        #logging.debug("Inside viterbi_iter")
        while len(nodes0) != 0:

            node = nodes0.pop()

            isfinal = False
            #TrellisHMMNode node
            sym = self.sg.get_state_info(node.s)[1]
            auxp = amodel.compute_gmm_emission_prob(fea, sym, node.q)
            #logging.debug(auxp)
            node.p+=auxp
            node.hmmp+=auxp

            p1_trans = amodel.get_transitions(sym,node.q)
            # This could be precomputed beforehand when loading the model...
            p0_trans = np.log(1  - np.exp(p1_trans))

            #logging.debug(p0_trans)
            #logging.debug(p1_trans)
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

        #logging.debug([str(x) for x in self.hmm_nodes1])
        
        # At this point, there could be nodes in sg_null_nodes1 and sg_nodes1
        # We should performe sg_null_nodes0 -> sg_nodes0 (viterbi_iter_sg)
        # and sg_nodes0 -> hmm_nodes1 (viterbi_sg2hmm)

        self.sg_null_nodes0 = self.sg_null_nodes1.copy()

        self.sg_null_nodes1 = []
    
        while len(self.sg_null_nodes0) != 0:
            self.expand_sg_nodes(self.sg_null_nodes0)
            #logging.debug(self.sg_null_nodes0)

        self.sg_nodes0 = self.sg_nodes1.copy()

        self.sg_nodes1 = []
        #logging.debug("Nodes0 list:")
        #logging.debug("\n".join([str(x) for x in self.sg_nodes0]))
        #logging.debug("Nodes1 list:")
        #logging.debug("\n".join([str(x) for x in self.sg_nodes1]))

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
                  logging.debug("Found: {}".format(n))
                  min_heap.pop(i)

          heapq.heappush(min_heap, node_id)
          return 


    def insert_hmm_node(self, hmmnode, fea):
        
        sym = self.sg.get_state_info(hmmnode.s)[1] # (state_id, symbol, word, edges_begin, edges_end)
        # Pruning could be done after or before getting the emission score
        auxp = amodel.compute_gmm_emission_prob(fea, sym, 0)
        prob = hmmnode.p + auxp # In which case this could be HUGE_VAL in the C implementation?
        
        hmm_actives = self.hmm_actives
        hmm_nodes1 = self.hmm_nodes1
        hmm_nodes1_heap = self.hmm_nodes1_heap

        if prob < self.v_thr:
            logging.debug("Pruned by v_thr")
            logging.debug("Node: {}, prob:{} v_thr: {}".format(hmmnode, prob, self.v_thr))
            return

        full = len(hmm_nodes1_heap) == self.nmaxhyp

        # If this hyp. is worse than the worst, we will prune it
        if full and prob <= hmm_nodes1_heap[0][0]:
            logging.debug("Pruned by min_heap")
            logging.debug("Node: {}, prob:{} min_prob: {}".format(hmmnode, prob, hmm_nodes1_heap[0][0]))
            return
        

        pos = hmm_actives.get(hmmnode.id, -1)

        if pos == -1: #New node
            
            logging.debug("New node")

            if prob > self.v_max:
                self.v_max = prob
                self.v_thr = prob - self.v_abeam
                self.v_maxh = hmmnode.hyp

            if not full:
                logging.debug("Nodes is not full")
                hmm_nodes1.append(hmmnode)
                cur_pos = len(hmm_nodes1) - 1

                self.heap_push(hmm_nodes1_heap, (prob, cur_pos), self.NEW)
            
                hmm_actives[hmmnode.id] = cur_pos  
                logging.debug("\n".join([str(x) for x in hmm_nodes1]))
                logging.debug("*****")
                logging.debug("\n".join([str(x) for x in hmm_actives]))
                logging.debug("*****")            
                logging.debug("\n".join([str(x) for x in hmm_nodes1_heap]))          
                logging.debug("=====")

            else:
                logging.debug("Nodes is full")
                logging.debug("Before")
                logging.debug("\n".join([str(x) for x in hmm_nodes1]))
                logging.debug("*****")
                logging.debug("\n".join([str(x) + " : " + str(hmm_actives[x]) for x in hmm_actives]))
                logging.debug("*****")
                logging.debug("\n".join([str(x) for x in hmm_nodes1_heap]))
                logging.debug("=====")
                #Remove an old node...
                _, pos_worse_node = heapq.heappop(hmm_nodes1_heap)

                hmm_actives[(hmm_nodes1[pos_worse_node].s,hmm_nodes1[pos_worse_node].q)] = -1
                #I am keeping the old node in the hmm_nodes1 list
                #If I remove the node, I should do something with the positions...

                #Adding the new one
                hmm_nodes1.append(hmmnode)
                cur_pos = len(hmm_nodes1) - 1
                self.heap_push(hmm_nodes1_heap, (prob, cur_pos), self.NEW)
                #heapq.heappush(hmm_nodes1_heap, (node.p, cur_pos))

                hmm_actives[hmmnode.id] = cur_pos
                logging.debug("After")
                logging.debug("\n".join([str(x) for x in hmm_nodes1]))
                logging.debug("*****")
                logging.debug("\n".join([str(x) + " : " + str(hmm_actives[x]) for x in hmm_actives]))
                logging.debug("*****")
                logging.debug("\n".join([str(x) for x in hmm_nodes1_heap]))
                logging.debug("=====")


        else: #Old node, should we update its value?
            
            logging.debug("Node is already inside")
            cur_node = hmm_nodes1[pos]
            
            if cur_node.p > prob:

                logging.debug("It is not better than the old node...")
                return

            else:

                if prob > self.v_max:
                    self.v_max = prob
                    self.v_thr = prob - self.v_abeam
                    self.v_maxh = hmmnode.hyp

                logging.debug("It is better than the old node, replacing...")
                cur_node.p = prob
                cur_node.hmmp = hmmnode.hmmp
                self.heap_push(hmm_nodes1_heap, (cur_node.p, pos), self.UPDATE)
                #heapq.heappush(hmm_nodes1_heap, (node.p, pos))        





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
            #logging.debug("Putting active in pos: {}".format(pos))
            self.actives[node.s] = pos
            # Added to null_nodes1/nodes1 depending on symbol node or not
            nodes1.append(node)
            #logging.debug([str(x) for x in nodes1])
            # If this is a word node, restart and manage hypothesis
            if insert_word:
                logging.debug(sg_state[2])
                node.hmmp = 0.0
                node.lmp = 0.0
        # Node already inserted, should be updated?
        else:
            #logging.debug("IT is active: {}".format(node))
            pos = self.actives[node.s]
            #logging.debug([str(x) for x in self.sg_nodes1])
            old_node = self.sg_nodes1[pos]
            #logging.debug("Replace?: {}".format(old_node))
            # Update stuff...
            score = old_node.p

            if node.p > score:
                self.sg_nodes1[pos].p = node.p
                #logging.debug("Yes")
                #logging.debug([str(x) for x in self.sg_nodes1])


    def decode(self, fea):
        
        
        fprob = 0.0

        self.viterbi_init(fea.sample[:,0])

        fprob += self.v_max

        logging.debug("Fprob: {}".format(fprob))
        

        # Apply adaptative beam?


        t_max = fea.sample.shape[1]

        t_fea = fea.sample[:,1]

        self.viterbi(t_fea)

        #for i in range(1,t_max):
        #for i in range(1,100):

        #    t_fea = fea.sample[:,i]

        #    self.viterbi(t_fea)


if __name__=="__main__":

   #logging.basicConfig(filename='deco.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
   logging.basicConfig(filename='deco.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

   import pickle
   amodel = pickle.load( open("../models/monophone_model_I32.p", "rb"))
   sg = pickle.load( open("../models/2.graph.p","rb"))

   from utils.TLSample import TLSample
   sample = TLSample("../samples/AAFA0016.features")
   decoder = Decoder(amodel, sg)

   decoder.decode(sample)
