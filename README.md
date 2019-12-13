# PyDecoder

Basic Speech Recognition Decoder implemented in Python.

## Requirements

* Python 3.7 
* Numpy

Not tested in versions <3.7, but it should work as well.

## Structure

* code: Python code 
* models: Models required to do the recognition
  * [Gaussian Mixture Acoustic Model](./models/monophone_model_I32) (plain and binary)
  * [Search Graph](./models/2.gram.graph) (plain and binary)
* samples: Example sample to recognise, with:
  * [WAV file](./samples/AAFA0016.wav)
  * [MFCC acoustic features](./samples/AAFA0016.features) (plain and binary)
  * [Text file with the transcription](./samples/AAFA0016.txt) (plain and binary)

## Run the code

To recognise the provided sample, run:

```
cd code
python -m decoder.Decoder
```

The output will be generated in the **deco.info.log** file.

## TO DO

* Include rescoring LM.
* Generate Wordgraphs.
* Advanced pruning.