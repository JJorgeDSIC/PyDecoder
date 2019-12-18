import sys
import numpy as np

class GaussianMixtureState(object):

    def __init__(self, symbol, dim, components):
        self.symbol = symbol
        self.dim = dim
        self.components = components

        self.pmembers = np.zeros((self.components,), dtype=np.float32)
        self.mus = np.zeros((self.components, self.dim), dtype=np.float32)
        self.ivars = np.zeros((self.components, self.dim), dtype=np.float32)

    def __str__(self):

        s = ""
        s+= "Symbol: {}, dim: {}, I:{}, ".format(self.symbol,self.dim, self.components)
        s+= "PMembers: {}, MU: {}, VAR:{}".format(self.pmembers,self.mus, self.ivars)
        return s
             
class GaussianMixtureTrans(object):

    def __init__(self, phoneme, num_hmm_states):
        self.phoneme = phoneme
        self.num_hmm_states= num_hmm_states

        self.trans = np.zeros((self.num_hmm_states,), dtype=np.float32)
        self.senones = []

    def __str__(self):

        s = ""
        s+= "Phoneme: {}, HMM states: {}".format(self.phoneme,self.num_hmm_states)
        s+= ", Trans: {}, Senones: {}".format(self.trans,self.senones)
        return s
       
class AModel(object):

    def __init__(self, model_type, model_path):

        if model_type != "TiedStates" and model_type != "Mixture":
            raise Exception("Not implemented yet")

        self.TIEDSTATES = 0
        self.MIXTURE = 1
        self.cache = {}
        
        self.gms_dict = {}
        self.trans_dict = {}
        self.load_model(model_type, model_path)

        self.reset_cache()

    def reset_cache(self):
        self.cache = {}

    def load_model(self, model_type, model_path):

        if model_type == "TiedStates":
            self.model_type = self.TIEDSTATES
            self.load_tiedstates_model(model_path)

        if model_type == "Mixture":
            self.model_type = self.MIXTURE
            self.load_mixture_model(model_path)
        
    def read_gaussian_mixture_trans(self, model_file):
        phoneme = model_file.readline().strip().replace("'","")
        line = model_file.readline().split()
        if line[0] != "Q":
            print("KO: Q expected")
        
        num_hmm_states = int(line[1])
        gmt = GaussianMixtureTrans(phoneme, num_hmm_states)
        
        line = model_file.readline().split()
        if line[0] == "Trans":
            line = model_file.readline().split()
            gmt.trans = [float(x) for x in line]
            line = model_file.readline().split()
            gmt.senones = [x.strip() for x in line]
            self.trans_dict[gmt.phoneme] = gmt
        elif line[0] == "TransP":
            gmt.trans = self.trans_dict[line[1]].trans
            line = model_file.readline().split()
            gmt.senones = [x.strip() for x in line]
            self.trans_dict[gmt.phoneme] = gmt
        else:
            print("KO: Trans or TransP expected")


    def compute_emission_prob(self, fea, sym, q):

        if self.model_type == self.TIEDSTATES:
            # WIP: Include cache here.
            return self.compute_gmm_emission_prob_tiedstate(fea, senone)

        elif self.model_type == self.MIXTURE:

            if (sym,q) in self.cache:
                return self.cache[(sym,q)]
            else:
                self.cache[(sym,q)] = self.compute_gmm_emission_prob_mixture(fea, sym, q)
                return self.cache[(sym,q)]
        else:
            raise Exception("Model unknown")

    def compute_gmm_emission_prob_tiedstate(self, fea, senone):
        
        mus = self.gms_dict[senone].mus
        ivars_ = self.gms_dict[senone].ivars
        pmembers = self.gms_dict[senone].pmembers
        logcs = self.gms_dict[senone].logcs

        logprob_per_components = self.compute_gaussian_members(fea, mus, ivars_, logcs, pmembers)

        return self.compute_gaussian_robust_addition(logprob_per_components)
    
    def compute_gmm_emission_prob_mixture(self, fea, sym, q):
        
        mus = self.gms_dict[sym][q].mus
        ivars_ = self.gms_dict[sym][q].ivars
        pmembers = self.gms_dict[sym][q].pmembers
        logcs = self.gms_dict[sym][q].logcs

        logprob_per_components = self.compute_gaussian_members(fea, mus, ivars_, logcs, pmembers)

        return self.compute_gaussian_robust_addition(logprob_per_components)
    
    def compute_gaussian_members(self, fea, mus, ivars_, logcs, pmembers):
        aux_full  = fea - mus
        aux_full_squared = aux_full**2
        ret_full = aux_full_squared * ivars_
        res = np.sum(ret_full, axis=1)
        
        return pmembers + (-0.5 * res + logcs)

    def get_transitions(self, phoneme, q):
        return self.trans_dict[phoneme].trans[q]

    def get_senone(self, phoneme, q):
        return self.trans_dict[phoneme].senones[q]

    def compute_gaussian_robust_addition(self, logprob_per_components):
        LOGEPS = -36.0437
        
        max_value = np.max(logprob_per_components)
        res = logprob_per_components - max_value
        #TO REVIEW: robust_addition = np.sum(np.where( res >= LOGEPS, np.exp(res), 0))
        robust_addition = np.sum(np.exp(res))
        return max_value + np.log(robust_addition)
    
    
    def read_gaussian_mixture_state_on_tiedstate(self, model_file):
        
        LOG2PI = 1.83787706641
        symbol = model_file.readline().strip()
        dim = self.features_dim
        line = model_file.readline().split()

        if line[0] != "I":
            print("KO: I expected")
        
        components = int(line[1])
        gms = GaussianMixtureState(symbol, dim, components)
        
        line = model_file.readline().split()
        if line[0] != "PMembers":
                print("KO: PMembers expected")

        gms.pmembers = np.array([float(x) for x in line[1:]])
        #print(gms.pmembers)

        line = model_file.readline().strip()
        if line != "Members":
                print("KO: Members expected")

        gms.mus = np.zeros((components, dim), dtype=np.float32)
        vars = np.zeros((components, dim), dtype=np.float32)
        gms.logcs = np.zeros((components,), dtype=np.float32)

        for i in range(components):
            line = model_file.readline().split()
            if line[0] != "MU":
                print("KO: MU expected")
            gms.mus[i,:]= np.array([float(x) for x in line[1:]])
            #print(gms.mus)
            line = model_file.readline().split()
            if line[0] != "VAR":
                print("KO: VAR expected")
            vars[i,:]= np.array([float(x) for x in line[1:]])
            #print(vars)

            aux = 0 
            for d in range(dim): 
                aux += np.log(vars[i,d]) 
            aux+= dim * LOG2PI 
            logc = -0.5 * aux 
            gms.logcs[i] = logc

        gms.ivars =  1.0/vars
        
        self.gms_dict[symbol] = gms

    def read_gaussian_mixture_state(self, model_file):
        
        LOG2PI = 1.83787706641
        phoneme = model_file.readline().strip().replace("'","")
        line = model_file.readline().split()
        if line[0] != "Q":
            print("KO: Q expected")
        
        num_hmm_states = int(line[1])
        gmt = GaussianMixtureTrans(phoneme, num_hmm_states)
        
        line = model_file.readline().split()
        if line[0] == "Trans":
            line = model_file.readline().split()
            gmt.trans = [float(x) for x in line]
            gmt.senones = ["s{}:{}".format(x+1,phoneme) for x in range(num_hmm_states)]
            self.trans_dict[gmt.phoneme] = gmt
        elif line[0] == "TransL":
            
            line = model_file.readline() #I
            value = float(model_file.readline().split()[1])
            gmt.trans = [value]
            gmt.senones = ["s{}:{}".format(x+1,phoneme) for x in range(num_hmm_states)]
            self.trans_dict[gmt.phoneme] = gmt
            for _ in range(6):
                model_file.readline()

        else:
            print("KO: Trans or TransL expected")

        dim = self.features_dim

        self.gms_dict[phoneme] = []



        for i in range(num_hmm_states):

            line = model_file.readline().split()
            
            if line[0] != "I":
                print("KO: I expected")
            
            components = int(line[1])
            gms = GaussianMixtureState(phoneme, dim, components)
            
            line = model_file.readline().split()
            if line[0] != "PMembers":
                    print("KO: PMembers expected")

            gms.pmembers = np.array([float(x) for x in line[1:]])
            #print(gms.pmembers)

            line = model_file.readline().strip()
            if line != "Members":
                    print("KO: Members expected")

            gms.mus = np.zeros((components, dim), dtype=np.float32)
            vars = np.zeros((components, dim), dtype=np.float32)
            gms.logcs = np.zeros((components,), dtype=np.float32)

            for i in range(components):
                line = model_file.readline().split()
                if line[0] != "MU":
                    print("KO: MU expected")
                gms.mus[i,:]= np.array([float(x) for x in line[1:]])
                #print(gms.mus)
                line = model_file.readline().split()
                if line[0] != "VAR":
                    print("KO: VAR expected")
                vars[i,:]= np.array([float(x) for x in line[1:]])
                #print(vars)

                aux = 0 
                for d in range(dim): 
                    aux += np.log(vars[i,d]) 
                aux+= dim * LOG2PI 
                logc = -0.5 * aux 
                gms.logcs[i] = logc

            gms.ivars =  1.0/vars
            self.gms_dict[phoneme].append(gms)

    def load_mixture_model(self, model_path):


        with open(model_path, "r") as model_file:
            line = model_file.readline()
            if line.strip() != "AMODEL":
                print("KO: AMODEL expected")

            line = model_file.readline()
            if line.strip() != "Mixture":
                print("KO: Mixture expected")


            line = model_file.readline()
            if line.strip() != "DGaussian":
                print("KO: DGaussian expected")
            
            line = model_file.readline().split()
            if not line[1].isdigit():
                print("KO: dimension should be a digit")

            self.features_dim = int(line[1])

            line = model_file.readline().split()
            if line[0] != "SMOOTH":
                print("KO: SMOOTH expected")

            self.smooth_values = np.array([float(x) for x in line[1:]])

            line = model_file.readline().split()
            
            if line[0] != "N":
                print("KO: N expected")

            self.num_labels = int(line[1])
            print("N: {}".format(self.num_labels))

            for n in range(self.num_labels):
                self.read_gaussian_mixture_state(model_file)

    def load_tiedstates_model(self, model_path):


        with open(model_path, "r") as model_file:
            line = model_file.readline()
            if line.strip() != "AMODEL":
                print("KO: AMODEL expected")
            

            line = model_file.readline()
            if line.strip() != "TiedStates":
                print("KO: TiedStates expected")


            line = model_file.readline()
            if line.strip() != "Mixture":
                print("KO: Mixture expected")


            line = model_file.readline()
            if line.strip() != "DGaussian":
                print("KO: DGaussian expected")


            
            line = model_file.readline().split()
            if not line[1].isdigit():
                print("KO: dimension should be a digit")

            self.features_dim = int(line[1])

    
            line = model_file.readline().split()
            if line[0] != "SMOOTH":
                print("KO: SMOOTH expected")

            self.smooth_values = np.array([float(x) for x in line[1:]])

            line = model_file.readline().split()
            
            if line[0] != "N":
                print("KO: N expected")

            self.num_labels = int(line[1])
            print("N: {}".format(self.num_labels))

            line = model_file.readline()

            if line.strip() != "States":
                print("KO: States expected")

            for n in range(self.num_labels):
                self.read_gaussian_mixture_state_on_tiedstate(model_file)

            line = model_file.readline().split()
            if line[0] != "N":
                print("KO: N expected")
            self.num_trans = int(line[1])
            print("Trans N: {}".format(self.num_trans))
            
            for t in range(self.num_trans):
                self.read_gaussian_mixture_trans(model_file)

if __name__=="__main__":

    amodel = AModel("Mixture", "../models/monophone_model_I32")
