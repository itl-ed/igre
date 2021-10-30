from typing import List, Dict, Union, Set
import random

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Bernoulli
from faiss import IndexFlatIP
import faiss.contrib.torch_utils # support for indexing torch tensors

from igre.logic import Entities, Denotations, Symbol
from igre.logic.syntax import AtomicSentence

class Grounder:

    def __init__(self,
                input_size: int,
                emb_size: int,
                supp_size: int,
                arity: int,
                init_entities: Entities,
                init_obs: Tensor,
                default: float = 0.5,
                ) -> None:

        """Interactive (Binary) Grounder

        :param input_size: length of the input vector
        :param emb_size: length of the embedding vector
        :param supp_size: support set size
        :param arity: size of the predicate aritu (e.g., 1,2)
        :param init_entities: list of initial entities
        :param init_obs: initial obs of size [num_obs,input_size]
        :param default: default label value in case of unnawerness.
        """
        self.input_size: int = input_size
        self.emb_size: int = emb_size
        self.supp_size: int = supp_size
        self.arity: int = arity
        self.default: float = default

        self.query_proj = nn.Sequential(
            nn.Linear(in_features=self.input_size,
                    out_features=self.emb_size),
            nn.ReLU()
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(in_features=self.input_size,
                    out_features=self.emb_size),
            nn.ReLU()
        )

        params =  list(self.query_proj.parameters())
        params += list(self.obs_proj.parameters())
        self.optim = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.BCELoss()
        # aquired entities (ints or tuples)
        self.entities: Denotations = init_entities
        # aquired symbols e.g. "red", "green", 
        self.symbols: List[Symbol] = []
        # aquired observations Tensor of size [num_obs, input_size]
        assert init_obs.shape[0] == self.num_obs 
        assert init_obs.shape[1] == self.input_size
        self.obs: Tensor = init_obs
        # embedding index for fast lookup
        self.embs: IndexFlatIP = IndexFlatIP(self.emb_size)
        with torch.no_grad():
            self.embs.add(self.encode_obs(init_obs))
        # aquired label Tensor of size [num_obs, vocab_size]
        self.labels: Tensor = torch.ones((self.num_obs, self.vocab_size))
        self.labels = self.default * self.labels

    def add_symbols(self, symbols: List[Symbol]) -> None:
        """add new symbols and expand the label space
        :param symbols: list of symbols that may contain alredy observed symbols
        """
        neologisms = list(set(symbols) - set(self.symbols))
        self.symbols += neologisms
        # expand labels in case of neologisms observed
        if len(neologisms) != 0:
            labels = self.default*torch.ones((self.num_obs, len(neologisms)))
            self.labels = torch.hstack((self.labels, labels))   

    def update_labels(self, labels: Dict[AtomicSentence, float]) -> None:
        """update the label values:
        :param labels: dictionary of labels an their new values the label 
        is of a form symbol(entitty) e.g. red(1) or below(1,1)"""

        for label, value in labels.items():
            symbol, entity  = label.name, label.terms
            if len(entity) == 1:
                entity = entity[0] 
            assert entity in self.entities, print(f"{entity} not in {self.entities}")
            
            # dont learn symbols not required to ground.
            if symbol not in self.symbols:
                continue

            entity_idx = self.entities.index(entity)
            symbol_idx = self.symbols.index(symbol)

            self.labels[entity_idx, symbol_idx] = value

    @property
    def num_obs(self) -> int:
        """number of observations encountered/stored by the model"""
        return len(self.entities)

    @property
    def vocab_size(self) -> int:
        """number of symbols aquired by the model"""
        return len(self.symbols)

    def add_obs(self,
                obs: Tensor,
                entities: List[Denotations]) -> None:
        """add new observations to the model
        :param obs: vector of size [len(entities), input_size]
        :param entities: list of entities (denotations)"""
        assert obs.shape[0] == len(entities)
        assert obs.shape[1] == self.input_size
        assert len(set(self.entities) & set(entities)) == 0, "repeated entities"
        
        # add new entities 
        self.entities += entities
        # add new observations
        self.obs = torch.vstack((self.obs, obs))
        # add new embeddings
        with torch.no_grad():
            self.embs.add(self.encode_obs(obs))
        # expand labels
        labels = self.default*torch.ones((len(entities), self.vocab_size))
        self.labels = torch.vstack((self.labels, labels))

    def encode_query(self, query: Tensor) -> Tensor:
        """encode query
        :param query: vector of size [1, input_size]
        
        :return vector of size [1, emb_size]"""
        assert query.shape[0] == 1 and query.shape[1] == self.input_size
        return nn.functional.normalize(self.query_proj(query), dim = 0)

    def encode_obs(self, obs: Tensor) -> Tensor:
        """encode observations:
        :param obs: vector of size [N, input_size]
        
        :return vector of size [N, emb_size]"""
        assert obs.shape[1] == self.input_size
        return nn.functional.normalize(self.obs_proj(obs), dim = 0)

    def predict_labels(self, 
                    query: Tensor,
                    obs: Tensor,
                    obs_labels: Tensor) -> Tensor:
        """Binary Matching Network prediction

        :param query: tensor of size [1, emb_size]
        :param obs: tensor of size [emb_size, support_size]
        :param obs_labels: tensor of size [support_size, vocab_size]
        
        :return tensor of size [1, vocab_size]"""
        torch.set_printoptions(precision=10)

        scores = torch.matmul(query, obs.T)
        weights = torch.softmax(scores, dim=1)
        pred_labels = torch.matmul(weights, obs_labels)
        
        print(f"weights: {weights},{weights.shape}|obs: {obs_labels},{obs_labels.shape}|labels: {pred_labels},{pred_labels.shape}")
        return pred_labels

    def support(self, query: Tensor) -> Tensor:
        """get support for the query
        :param query: query vector of size [1, emb_size]

        :return idx of support in embs and support of size [supp_size, emb_size]
        """
        assert query.shape[0] == 1
        assert query.shape[1] == self.emb_size
        assert self.supp_size <= self.embs.ntotal

        _, idx, supp = self.embs.search_and_reconstruct(query, self.supp_size)

        #[supp_size],[supp_size,emb_size] <- [1,supp_size][1,supp_size,emb_size]
        idx, supp = idx.squeeze(0), supp.squeeze(0) 

        assert supp.shape[0] == self.supp_size
        assert supp.shape[1] == self.emb_size
        assert idx.shape[0] == self.supp_size 

        return idx, supp

    def predict(self, query: Tensor) -> Dict[str,float]:
        """Prediction at test time
        :param query: query vector of size [1, input_size]
        
        :return dict with key: symbols, value: prob of it being true for a query
        """
        assert query.shape[0] == 1 and query.shape[1] == self.input_size

        if len(self.symbols) == 0:
            return dict()

        with torch.no_grad():
            query_emb = self.encode_query(query) 
            idx, supp = self.support(query_emb)

            # [supp_size, vocab_size]
            supp_labels = self.labels[idx]
            # [1, vocab_size] 
            pred_labels = self.predict_labels(query_emb,supp,supp_labels)
            # format labels 
            pred_labels = [float(label) for label in pred_labels.squeeze(0)]

        return dict(zip(self.symbols,pred_labels))

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
        torch.set_printoptions(precision=10)

        print(f'current vocab: {self.symbols}')

        ## create a dataset 
        dataset = TensorDataset(self.obs, self.labels)
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
                obs_idx = list(set(range(0,batch_size)) - set([query_idx]))

                # [batch_size-1, input_size]
                obs: Tensor = data[obs_idx]
                # [batch_size-1, vocab_size]
                obs_labels: Tensor = labels[obs_idx]
                # [1, input_size]
                query: Tensor  = data[query_idx].unsqueeze(0)
                # [1, vocab_size]
                query_labels: Tensor = labels[query_idx]
                # [1, emb_size] <- [1, input_size]
                query = self.encode_query(query)
                # [batch_size-1, emb_size] <- [batch_size-1, input_size]
                obs = self.encode_obs(obs)
                # [1, vocab_size]
                print(f"query: {query.shape}")
                print(f"obs: {obs.shape}")
                print(f"labels: {obs_labels.shape}")
                pred_labels: Tensor = self.predict_labels(query,obs,obs_labels)
                print(f"pred labels: {pred_labels.shape}")
                # [vocab_size] <- [1,vocab_size]
                pred_labels = pred_labels.squeeze(0)
                print(f"prediction: {pred_labels}")
                
                pred_ent = Bernoulli(pred_labels).entropy()
                # mask = (pred_ent <= 0.55).type(torch.float32)
                pred_labels = pred_labels[pred_ent <= 0.55]
                query_labels = query_labels[pred_ent <= 0.55]
                print(f"pred_labels: {pred_labels} ", f"query_labels: {query_labels}")

                loss = self.criterion(pred_labels, query_labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.detach()

                iter += 1
                if iter % report_freq == 0:
                    print(f"[loss]{total_loss/report_freq}")
                    total_loss = 0

        ## update embs
        with torch.no_grad():
            embs = self.encode_obs(self.obs)
            self.embs = IndexFlatIP(self.emb_size)
            self.embs.add(embs)


