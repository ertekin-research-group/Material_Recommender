# Material_Recommender
Leveraging representations extracted from language models pretrained on material science literature for material discovery and property prediction.


## Framework
<p align="center">
  <img src="https://github.com/ertekin-research-group/Material_Recommender/blob/main/workflow.PNG" /width="1000"> 
</p>

## Installation and prerequisites
- To install the dependencies via Anaconda:
1. Create conda virtual env: `conda env create -f environment.yml`
2. Activate virtual env: `conda activate mat_rec`

- Download composition and structure embeddings:
The embeddings for 116K materials obtained in this work can be found [here](https://doi.org/10.6084/m9.figshare.22718668.v1)
`composition_embeddings_116k.h5`: embeddings on material compositions.
`structure_embeddings_116k.h5`: sentence embeddings on automatically generated material descriptions. Place the downloaded files under the main directory.

- Download pretrained weights:
Follow the instructions on [MatBERT repo](https://github.com/lbnlp/MatBERT) to download pretrained weights and tokenizer for the uncased model. Place the folder under matbert_model_fiels directory.



## Usage


