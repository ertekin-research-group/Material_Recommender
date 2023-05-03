# Material_Recommender
Leveraging representations extracted from language models pretrained on material science literature for material discovery and property prediction.

**(Code still under construction)**

## Framework
<p align="center">
  <img src="https://github.com/ertekin-research-group/Material_Recommender/blob/main/workflow.PNG" /width="1200"> 
</p>

## Installation and prerequisites
- To install the dependencies via Anaconda:
1. Clone the repo: `git clone https://github.com/ertekin-research-group/Material_Recommender.git`
3. Create conda virtual env: `conda env create -f environment.yml`
4. Activate virtual env: `conda activate matrec`

- Download composition and structure embeddings:
The embeddings for 116K materials obtained in this work can be found [here](https://doi.org/10.6084/m9.figshare.22718668.v1)
`composition_embeddings_116k.h5`: embeddings on material compositions.
`structure_embeddings_116k.h5`: sentence embeddings on automatically generated material descriptions. Place the downloaded files under the main directory.

- Download pretrained weights:
Follow the instructions on [MatBERT repo](https://github.com/lbnlp/MatBERT) to download pretrained weights and tokenizer for **the uncased model**. Place the folder under matbert_model_files directory.



## Usage
- Search material candidates in the representation space.<BR>
`TODO`
- Ranking candidates for materials with similar TE performance.<BR>
`TODO`
- Training MMoeE models on material representations.<BR>
`TODO`

## Cite
```
@misc{qu2023leveraging,
      title={Leveraging Language Representation for Material Recommendation, Ranking, and Exploration},
      author={Jiaxing Qu and Yuxuan Richard Xie and Elif Ertekin},
      year={2023},
      eprint={2305.01101},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```
