from typing import List, Dict
import random
from copy import deepcopy

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Bernoulli
from sklearn.neighbors import BallTree

from igre.logic.model import Denotation, Symbol
from igre.logic.syntax import AtomicSentence

class Grounder:

    def __init__(self,
                input_size: int,
                emb_size: int,
                supp_size: int,
                arity: int,
                default: float = 0.5,
                threshold: float = 0.7,
                ) -> None:

        """Interactive (Binary) Grounder

        :param input_size: length of the input vector
        :param emb_size: length of the embedding vector
        :param supp_size: support set size
        :param arity: size of the predicate aritu (e.g., 1,2)
        :param default: default label value in case of unnawerness.
        :param threshold: threshold used when deciding if evidence should be used.
        """
        self.input_size: int = input_size
        self.emb_size: int = emb_size
        self.supp_size: int = supp_size
        self.arity: int = arity
        self.default: float = default
        self.threshold: float = threshold

        self.query_proj = nn.Sequential(
            nn.Linear(in_features=self.input_size,
                    out_features=self.emb_size),
            nn.ReLU())

        self.key_proj = nn.Sequential(
            nn.Linear(in_features=self.input_size,
                    out_features=self.emb_size),
            nn.ReLU())

        p = list(self.query_proj.parameters())+list(self.key_proj.parameters())
        self.optim = torch.optim.Adam(p, lr=1e-4, weight_decay=1e-5)
        self.criterion = nn.BCELoss()
        # dictionary of denotation to key (features)
        self.keys: Dict[Denotation, Tensor] = dict()
        # dictionary of denotation to concept vectors
        self.values: Dict[Denotation, Tensor] = dict()
        # list of symbols
        self.symbols: List[Symbol] = []

    @property
    def num_denotations(self):
        assert len(self.keys) == len(self.values)
        return len(self.keys)

    @property        
    def vocab_size(self):
        return len(self.symbols)
                
    def update_values(self, atoms: Dict[AtomicSentence,float]) -> None:
        """update atom values
        :param atoms: dictionary of AtomicSentence::estimated semantic value"""
        assert all(atom.arity == self.arity for atom in atoms.keys())

        for atom, value in atoms.items():
            denotation = atom.terms[0] if self.arity == 1 else atom.terms
            self.values[denotation][:,self.symbols.index(atom.name)] = value

    def add_symbol(self, symbol: Symbol) -> None:
        """add new symbol"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            for d in self.values.keys():
                value = torch.hstack((self.values[d], Tensor([[self.default]])))
                self.values[d] = value

    def add_symbols(self, symbols: List[Symbol]) -> None:
        """add new symbols"""
        for symbol in symbols:
            self.add_symbol(symbol)

    def add_key(self, denotation: Denotation, feature: Tensor) -> None:
        """Add new key
        :param term: denotation for observation (index) 
        :param features: feature vector for observation"""
        assert denotation not in self.keys.keys()
        assert denotation not in self.values.keys()

        self.keys[denotation] = feature
        self.values[denotation] = Tensor([self.vocab_size*[self.default]])

    def add_keys(self, features: Dict[Denotation, Tensor]) -> None:
        """Add dictionairy for keys"""
        for denotation, feature in features.items():
            self.add_key(denotation, feature)
        
    def encode_query(self, query: Tensor) -> Tensor:
        """encode query
        :param query: vector of size [1, input_size]
        
        :return vector of size [1, emb_size]"""
        assert query.shape[0] == 1 and query.shape[1] == self.input_size
        return F.normalize(self.query_proj(query), dim = 0)

    def encode_keys(self, keys: Tensor) -> Tensor:
        """encode keys:
        :param obs: vector of size [N, input_size]
        
        :return vector of size [N, emb_size]"""
        assert keys.shape[1] == self.input_size
        return F.normalize(self.key_proj(keys), dim = 0)

    def attention(self, query: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """Attention kernel

        :param query: tensor of size [1, emb_size]
        :param keys: tensor of size [N, emb_size]
        :param values: tensor of size [N, vocab_size]
        
        :return tensor of size [1, vocab_size]"""

        assert query.shape[0] == 1 
        assert query.shape[1] == self.emb_size
        assert keys.shape[1] == self.emb_size 
        assert values.shape[1] == self.vocab_size

        scores = torch.matmul(query, keys.T)
        weights = torch.softmax(scores, dim=1)
        attn = torch.matmul(weights, values)
        
        return attn

    def __get_support_denotations(self,
                            query: Tensor,
                            supp: Dict[Denotation,Tensor]) -> List[Denotation]:
        """Get denotations that can support query
        :param query: tensor of size [1, emb_size]
        :param supp: dict of denotation to tensors of size [1, input_size]

        return List of denotations
        """
        assert query.shape[0] == 1
        assert query.shape[1] == self.emb_size
        assert all([f.shape[0] == 1 for f in supp.values()])
        assert all([f.shape[1] == self.input_size for f in supp.values()])

        denotations, features = list(supp.keys()), list(supp.values())
        embs = self.encode_keys(torch.vstack(features))
        idxs = BallTree(embs).query(query,return_distance=False, k=self.supp_size)[0]

        return [denotations[idx] for idx in idxs ]
        
    def __get_support_values(self,
                            values: Dict[Denotation, Tensor],
                            denotations: List[Denotation]) -> Tensor:
        """ Get value matrix
        :param value: denotation::concept vector dictionary
        :param denotations: list of denotations to use

        :return tensor of size [len(denotations), vocab_size]
        """
        values = torch.vstack([v for k,v in values.items() if k in denotations])

        assert values.shape[0] == len(denotations)
        assert values.shape[1] == self.vocab_size

        return values

    def __get_support_keys(self, 
                    keys: Dict[Denotation, Tensor],
                    denotations: List[Denotation]) -> Tensor:
        """ Get keys
        :param query: query vector of size [1, emb_size]
        :param denotations: denotations from which aim tobuld support  

        :return [len(denotations), emb_size]
        """
        keys = torch.vstack([v for k,v in keys.items() if k in denotations])
        keys = self.encode_keys(keys)

        assert keys.shape[0] == len(denotations)
        assert keys.shape[1] == self.emb_size

        return keys

    def predict(self, query: Tensor) -> Tensor:
        """Concept prediction given a query
        :param query: query vector of size [1, input_size]
        
        :return tensor of size [1, vocab_size]
        """
        assert query.shape[0] == 1 and query.shape[1] == self.input_size

        with torch.no_grad():

            query = self.encode_query(query)

            denotations = self.__get_support_denotations(query, self.keys)

            keys = self.__get_support_keys(self.keys, denotations)

            values = self.__get_support_values(self.values, denotations)


        return self.attention(query, keys, values)

    def batch_learning_mode(self,
                        epochs: int,
                        batch_size: int,
                        shuffle: bool,
                        report_freq: int) -> None:
        """Batch learning mode to update model parameters
        :param epochs: number of epochs
        :param batch_size: size of the minibatch
        :param num_queries: how many queries to draw from a batch
        :param shuffle: if True, shuffle batches
        :param report_freq: frequency of reporting
        """
        
        # print("=========================")
        # print(f"logging dataset used")
        # for denotation, concepts in self.values.items():
        #     print(f"denotation: {denotation}")
        #     concepts_dict = {k:float(v) for k,v in zip(self.symbols,concepts.flatten())}
        #     print(f"concepts: {concepts_dict}")

        keys = torch.vstack(list(self.keys.values()))
        values = torch.vstack(list(self.values.values()))
        
        assert keys.shape[0] == self.num_denotations
        assert keys.shape[1] == self.input_size
        assert values.shape[0] == self.num_denotations
        assert values.shape[1] == self.vocab_size

        dataset = TensorDataset(keys, values)

        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)
        iter = 0
        total_loss = 0
        for _ in range(epochs):

            for data, labels in loader:

                batch_size = data.shape[0]
                
                # at least two elements required
                if batch_size < 2:
                    continue

                query_idx = random.randint(0, batch_size-1)
                key_idxs = list(set(range(0,batch_size)) - set([query_idx]))

                # [batch_size-1, input_size]
                keys: Tensor = data[key_idxs]
                # [batch_size-1, vocab_size]
                values: Tensor = labels[key_idxs]
                # [1, input_size]
                query: Tensor  = data[query_idx].unsqueeze(0)
                # [1, vocab_size]
                values_true: Tensor = labels[query_idx]
                # [1, emb_size] <- [1, input_size]
                query = self.encode_query(query)
                # [batch_size-1, emb_size] <- [batch_size-1, input_size]
                keys = self.encode_keys(keys)
                # [1, vocab_size]
                values_pred: Tensor = self.attention(query,keys,values)
                # [vocab_size] <- [1,vocab_size]
                values_pred = values_pred.squeeze(0)
                # play safe
                values_pred = torch.clamp(values_pred, min=0.0, max=1.0)                
                pred_ent = Bernoulli(values_pred).entropy()
                # mask = (pred_ent <= 0.55).type(torch.float32)
                values_pred = values_pred[pred_ent <= self.threshold]
                values_true = values_true[pred_ent <= self.threshold]

                loss = self.criterion(values_pred, values_true)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.detach()

                iter += 1
                if iter % report_freq == 0:
                    print(f"[loss]{total_loss/report_freq}")
                    total_loss = 0
