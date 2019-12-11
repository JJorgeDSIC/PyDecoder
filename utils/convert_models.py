from code.AModel import AModel
from code.SGraphModel import SGraphModel
import pickle

amodel = AModel("Mixture", "models/monophone_model_I32")
sg = SGraphModel("models/2.gram.graph")

pickle.dump( sg, open( "models/2.graph.p", "wb" ) )
pickle.dump( amodel, open( "models/monophone_model_I32.p", "wb" ) )
