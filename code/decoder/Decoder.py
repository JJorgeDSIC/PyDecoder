import numpy as np
from models.AModel import AModel
from models.SGraphModel import SGraphModel
import heapq
import logging

def print_lst(lst):
    logging.debug("HMM nodes list")
    logging.debug("\n".join([str((x.p, x.id)) for x in lst]))

def print_lst_sg(lst):
    logging.debug("SG nodes list")
    logging.debug("\n".join([str(x) for x in lst]))

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

    
    class wordHyp(object):
        def __init__(self, prev, word):
            self.prev = prev
            self.word = word

        def __str__(self):
            return "Prev: {}, word: {}".format(self.prev, self.word)

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
            return "S: {}, Q: {}, ID: {}, P: {:.5f} HMMP: {:.5f}, L.WORD: {}".format(self.s, self.q, self.id, self.p, self.hmmp, self.hyp)

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
        self.beam = 2**10
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
        self.nmaxhyp = 20

        # Enums
        self.NEW = 0
        self.UPDATE = 1


    def expand_sg_nodes(self, node_lst):

        self.max_prob = -2**10
        self.max_node = None

        while len(node_lst) != 0:

            node = node_lst.pop()
            sg_state = self.sg.get_state_info(node.s)

            if self.final_iter and node.s ==  self.sg.final:
                if node.p > self.max_prob:
                    self.max_prob = node.p
                    self.max_node = node
                    self.max_hyp = node.hyp

            current_p = node.p
            current_lmp = node.lmp
            prev_s = node.s
            edges_begin = sg.edges_begin[prev_s]
            edges_end = sg.edges_end[prev_s]

            for pos in range(edges_begin, edges_end):
                s = sg.edges_dst[pos]
                if not self.final_iter:
                    if sg.edges_dst[pos] == sg.final:
                        # logging.info("Final reached")
                        continue
                else:
                    # We want to arrive to the final state
                    sym = self.sg.get_state_info(s)[1] # (state_id, symbol, word, edges_begin, edges_end)
                    if sym != '-':
                        continue

                p = current_p + sg.edges_weight[pos] * self.GSF + self.WIP
                lmp = current_lmp + sg.edges_weight[pos]
                sgnode = self.TrellisSGNode(s, p, 0, lmp, None, node.hyp)
                self.insert_sg_node(sgnode)

    def viterbi_init(self, fea):

        #Initialize the search
        self.current_fea = fea

        self.amodel.reset_cache()

        # Empty hyp
        first_hyp = 0

        sgnode = self.TrellisSGNode(self.sg.start, 0.0, 0.0, 0, None, first_hyp)
        self.actives = [-1] * self.sg.nstates
        self.actives[self.sg.start] = 0
        self.sg_null_nodes0 = []
        self.sg_null_nodes0.append(sgnode)

        self.iter_sg_init()
        
        # From the worst to the best node, reversed
        while(len(self.hmm_nodes1_heap) != 0):
            prob, pos = heapq.heappop(self.hmm_nodes1_heap)
            self.hmm_nodes0.append(self.hmm_nodes1[pos])

        self.hmm_nodes1 = []
        self.hmm_nodes1_heap = []
        self.hmm_actives = {}

    def iter_sg_init(self):

        # Expand null_nodes0 -> null_nodes1
        self.expand_sg_nodes(self.sg_null_nodes0)

        # Expand null_nodes0 -> null_nodes1
        # and    null_nodes0 -> nodes1
        while len(self.sg_null_nodes1) != 0:
            # Workaround, TO DO, to review this step...
            for node in self.sg_null_nodes1:
                self.actives[node.s] = -1
                self.sg_null_nodes0.append(node)
            
            self.sg_null_nodes1 = []
            self.expand_sg_nodes(self.sg_null_nodes0)

        # null_nodes1 is empty now, it contained the SG nodes w/o syms or words
        # nodes1 contains the SG nodes that have syms or words associated
        self.copy_sg_lst(self.sg_nodes1, self.sg_nodes0)
        self.sg_nodes1 = []
        # From SG nodes to HMM nodes
        self.from_sg_to_hmm_nodes()
    

    def copy_sg_lst(self, src, dst):
        
        for node in src:
            self.actives[node.s] = -1
            dst.append(node)

    def from_sg_to_hmm_nodes(self):

        for node in self.sg_nodes0:
           sym = self.sg.get_state_info(node.s)[1] # (state_id, symbol, word, edges_begin, edges_end)
           # TO DO: Rethink this array-like representation...
           edges_begin = self.sg.edges_begin[node.s]
           edges_end = self.sg.edges_end[node.s]
           for pos in range(edges_begin, edges_end):
               hmmnode = self.TrellisHMMNode(node.s, 0, node.p, node.hmmp, node.lmp, node.lm_state, node.hyp)
               self.insert_hmm_node(hmmnode, self.current_fea)

    def iter_sg(self):

        # TO DO: Refactor, shallow or deep copy? maybe they will contain more complex info
        # sg_nodes0 <- sg_nodes1
        self.copy_sg_lst(self.sg_nodes1, self.sg_nodes0)
        self.sg_nodes1 = []

        # Expand sg_nodes0 -> sg_nodes1
        if len(self.sg_nodes0) != 0:
            self.expand_sg_nodes(self.sg_nodes0)

        # Expand sg_null_nodes0 -> sg_null_nodes1
        # New sg_null_nodes could appear, we should iterate until there are none
        while len(self.sg_null_nodes1) != 0:

            # Workaround, TO DO, to review this step...
            self.copy_sg_lst(self.sg_null_nodes1, self.sg_null_nodes0)
            self.sg_null_nodes1 = []
            self.expand_sg_nodes(self.sg_null_nodes0)
        
        # TO DO: Refactor this part
        # null_nodes1 is empty now, it contained the SG nodes w/o syms or words
        # nodes1 contains the SG nodes that have syms or words associated
        self.copy_sg_lst(self.sg_nodes1, self.sg_nodes0)
        self.sg_nodes1 = []
        # From SG nodes to HMM nodes
        self.from_sg_to_hmm_nodes()

    def viterbi(self, fea, t):

        self.t = t

        self.current_fea = fea

        self.amodel.reset_cache()

        self.sg_nodes0 = []
        self.sg_nodes1 = []
        self.actives = [-1] * self.sg.nstates

        hmm_nodes0 = self.hmm_nodes0

        old_max = self.v_max
        old_thr = self.v_thr
        self.v_max = -2**10
        self.v_thr = -2**10
        self.v_lm_max = -2**10
        self.v_lm_thr = -2**10
        self.v_abeam = self.beam

        while len(hmm_nodes0) != 0:

            node = hmm_nodes0.pop()
            logging.debug("HE")
            logging.debug(node)

            if node.p < old_thr:
                continue
            
            node.p -= old_max

            isfinal = False
            #TrellisHMMNode node
            sym = self.sg.get_state_info(node.s)[1]
            """
            another strategy: emit before expanding the state
            auxp = amodel.compute_emission_prob(fea, sym, node.q)
            #logging.debug(auxp)
            node.p+=auxp
            node.hmmp+=auxp
            """
            p1_trans = amodel.get_transitions(sym,node.q)
            # This could be precomputed beforehand when loading the model...
            p0_trans = np.log(1  - np.exp(p1_trans))

            current_p = node.p
            current_hmmp = node.hmmp

            #Staying in the same state, I could reuse the old node, TO DO
            hmmnode = self.TrellisHMMNode(node.s, node.q, current_p + p0_trans, current_hmmp + p0_trans, node.lmp, node.lm_state, node.hyp)
            if not self.final_iter:
                self.insert_hmm_node(hmmnode, fea)

            if node.q + 1 < amodel.trans_dict[sym].num_hmm_states:

                #Jumping to the following state
                hmmnode = self.TrellisHMMNode(node.s, node.q + 1, current_p + p1_trans, current_hmmp + p1_trans, node.lmp, node.lm_state, node.hyp)
                if not self.final_iter:
                     self.insert_hmm_node(hmmnode, fea) # from 0->1, 1->2

            else: #from 2-> outside
                isfinal = True

            if isfinal:
                sgnode = self.TrellisSGNode(node.s, node.p + p1_trans, node.hmmp + p1_trans, node.lmp, None, node.hyp)
                #if self.final_iter:
                #    logging.info("Final expansion: {}".format(sgnode))
                self.insert_sg_node(sgnode)

        # After this step, SG nodes in null_nodes1 or sg_nodes1 have been created
        # We should do the iteration also for SG nodes
        self.iter_sg()
      
        # hmm_nodes0 <- hmm_nodes1
        self.hmm_nodes0 = []
        
        # From the worst to the best node, reversed
        while(len(self.hmm_nodes1_heap) != 0):
            prob, pos = heapq.heappop(self.hmm_nodes1_heap)
            self.hmm_nodes0.append(self.hmm_nodes1[pos])

        self.hmm_nodes1 = []
        self.hmm_actives = {}


    def update_heap(self, min_heap, node_id):
        #Workaround to not implement my own heap...
        pos = 0
        while min_heap[pos][1] != node_id[1]:
            pos+=1

        assert(pos != len(min_heap))

        min_heap[pos] = node_id

        #heapq.heappush(min_heap, node_id)
        heapq.heapify(min_heap)

    def heap_push(self, min_heap, node_id, mode):
    
        if mode == self.NEW:
            heapq.heappush(min_heap, node_id)
            return

        if mode == self.UPDATE:
          self.update_heap(min_heap, node_id)
          return 

    def insert_hmm_node(self, hmmnode, fea):
        
        sym = self.sg.get_state_info(hmmnode.s)[1] # (state_id, symbol, word, edges_begin, edges_end)
        # Pruning could be done after or before getting the emission score
        auxp = amodel.compute_emission_prob(fea, sym, hmmnode.q)
        prob = hmmnode.p + auxp # In which case this could be HUGE_VAL in the C implementation?
        hmmnode.p = prob
        hmmnode.hmmp += auxp
        
        hmm_actives = self.hmm_actives
        hmm_nodes1 = self.hmm_nodes1
        hmm_nodes1_heap = self.hmm_nodes1_heap

        if prob < self.v_thr:
            return

        full = len(hmm_nodes1_heap) == self.nmaxhyp

        # If this hyp. is worse than the worst, we will prune it
        if full and prob <= hmm_nodes1_heap[0][0]:
            return
        
        pos = hmm_actives.get(hmmnode.id, -1)
        
        if pos == -1: #New node
        
            if prob > self.v_max:
                self.v_max = prob
                self.v_thr = prob - self.v_abeam
                self.v_maxh = hmmnode.hyp

            if not full:
                hmm_nodes1.append(hmmnode)
                cur_pos = len(hmm_nodes1) - 1

                self.heap_push(hmm_nodes1_heap, (prob, cur_pos), self.NEW)
            
                hmm_actives[hmmnode.id] = cur_pos  
            else:
                #Remove an old node...
                _, pos_worse_node = heapq.heappop(hmm_nodes1_heap)

                hmm_actives[(hmm_nodes1[pos_worse_node].s,hmm_nodes1[pos_worse_node].q)] = -1
                #I am keeping the old node in the hmm_nodes1 list
                #If I remove the node, I should do something with the positions...

                #Adding the new one
                hmm_nodes1.append(hmmnode)
                cur_pos = len(hmm_nodes1) - 1
                self.heap_push(hmm_nodes1_heap, (prob, cur_pos), self.NEW)

                hmm_actives[hmmnode.id] = cur_pos

        else: #Old node, should we update its value?
            
            cur_node = hmm_nodes1[pos]

            if cur_node.p > prob:
                return
            else:
                if prob > self.v_max:
                    
                    self.v_max = prob
                    self.v_thr = prob - self.v_abeam
                    self.v_maxh = hmmnode.hyp

                cur_node.p = prob
                cur_node.hmmp = hmmnode.hmmp
                self.heap_push(hmm_nodes1_heap, (cur_node.p, pos), self.UPDATE)

    def update_lm_beam(self, value):

        self.v_lm_max = value
        self.v_lm_thr = value - self.v_lm_beam

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
            if node.p > self.v_lm_max:
                self.update_lm_beam(node.p)

            pos = len(nodes1)
            # Added to actives
            self.actives[node.s] = pos
            # Added to null_nodes1/nodes1 depending on symbol node or not
            nodes1.append(node)
            # If this is a word node, restart and manage hypothesis
            if insert_word:
                hyp = self.wordHyp(node.hyp, sg_state[2])
                self.hyp_lst.append(hyp)
                node.hyp = len(self.hyp_lst) - 1
                node.hmmp = 0.0
                node.lmp = 0.0

        # Node already inserted, should be updated?
        else:
            pos = self.actives[node.s]
            old_node = nodes1[pos]
            
            # If this node is better
            if node.p >  old_node.p: 
                if node.p > self.v_lm_max:
                    self.update_lm_beam(node.p)
                # Word node
                if insert_word:
                    # Do word-related stuff, TO DO    
                    # Update fields
                    nodes1[pos].p = node.p
                    nodes1[pos].hyp = node.hyp
                # No-Word node
                else:
                    nodes1[pos] = node
    
    def get_max_hyp(self):

        output = [self.hyp_lst[self.max_hyp].word]
        prev = self.hyp_lst[self.max_hyp].prev
        while prev != 0:
            output.append(self.hyp_lst[prev].word)
            prev = self.hyp_lst[prev].prev

        return (" ".join([x for x in reversed(output)]))

    def decode(self, fea):
        
        fprob = 0.0

        self.hyp_lst = []

        self.hyp_lst.append(self.wordHyp(-1,""))

        self.final_iter = False

        #For the first approach I will increment the list
        # the next iteration I will use another structure to manage
        # the allocation/release stuff to reuse some space

        self.viterbi_init(fea.sample[:,0])
        fprob += self.v_max
        
        # Apply adaptative beam?
        t_max = fea.sample.shape[1]

        for i in range(1,t_max):
            t_fea = fea.sample[:,i]
            self.viterbi(t_fea,i)
            
        self.final_iter = True
        self.viterbi(None,i)

        sentence = self.get_max_hyp()

        logging.info("Recognised: {}".format(sentence))

if __name__=="__main__":

   logging.basicConfig(filename='deco.info.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
   #logging.basicConfig(filename='deco.debug.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

   import pickle
   amodel = pickle.load( open("../models/monophone_model_I32.p", "rb"))
   sg = pickle.load( open("../models/2.graph.p","rb"))

   from utils.TLSample import TLSample
   sample = TLSample("../samples/AAFA0016.features")
   decoder = Decoder(amodel, sg)
   decoder.nmaxhyp = 20
   decoder.decode(sample)
