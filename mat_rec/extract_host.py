ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
            "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
            "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
            "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
            "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition

robo_data=pd.read_pickle('robo_descriptions.pkl')

element_stoi_all = []
for composition in robo_data['composition_name']:
    element_stoi = np.array([*map(Composition(composition).get_el_amt_dict().get, ELEMENTS)])
    element_stoi[element_stoi==None] = 0
    element_stoi_all.append(element_stoi)

A_sparse = sparse.csr_matrix(np.array(element_stoi_all).astype(float))

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import sparse

def get_host(composition,sparse_stoi,composition_names):

    element_stoi = np.array([*map(Composition(composition).get_el_amt_dict().get, ELEMENTS)])
    element_stoi[element_stoi==None] = 0
    similarities = cosine_similarity(np.array(element_stoi).reshape(1, -1), sparse_stoi)

    return composition_names[np.argsort(similarities[0])[::-1][0]]
