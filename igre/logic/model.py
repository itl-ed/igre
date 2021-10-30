"""
Model-theoretic semantic interpretation 
for predicate logic with generalized quantifiers
"""
from __future__ import annotations
from typing import Dict, Tuple, Union, Set, List
from collections import defaultdict
from itertools import product
from copy import deepcopy as copy
from functools import reduce
from .syntax import (Grammar, RefExp, Sentence, AtomicSentence, NegatedSentence, 
            AndSentence, OrSentence, ImpSentence, IffSentence, 
            QuantifierSentence, get_nums, Variable, Symbol)

# Complex types
Entity = int
Entities = Set[int]
Denotation = Union[int,Tuple[int,int]]
Denotations = Set[Denotation]


class DomainModel:

    def __init__(self, 
                entities: Entities = set(), 
                extension: Dict[Symbol,Denotations] = defaultdict(set),
                entity_symbol: str = 'object',
                grammar: Grammar = Grammar()) -> None:
        """Domain Model

        :param entities: set of entities, represented by integers: 1,2,..
        :param extension: extension function mapping non-logical symbols 
        to denotations

        """
        self.entities = entities
        self.extension = extension
        self.entity_symbol = entity_symbol
        # add object preperty for all domain entities
        self.extension[self.entity_symbol] = entities
        self.grm = grammar


    def __str__(self) -> str:
        return f"""ENTITIES:{self.entities}
        EXTENTION:{dict.__repr__(self.extension)}"""

    def reindex(self, start: int) -> Tuple[int,Dict[Denotation,Denotation]]:
        """reindex domain
        :param start: new start index
        
        :return start of index, entity reindex dictionary"""
        prop = {e:e+start for e in self.entities}
        rel = {es: tuple(e+start for e in es) for es in product(self.entities, repeat=2)}
        index = {**prop, **rel}

        # reindex entities
        self.entities = set(e+start for e in self.entities)
        # reindex extension 
        for pred, terms in self.extension.items():
            new_terms = set()
            for term in terms:
                if isinstance(term, int):
                    new_terms.add(term+start)
                elif isinstance(term, tuple):
                    new_terms.add(tuple(t+start for t in term))
                else:
                    Exception("not supported")

            self.extension[pred] = new_terms

        start += len(self.entities)
        return start, index

    def invert(self) -> defaultdict:
        """Get model in inverted format (denotation - list of symbols)"""
        symbols = defaultdict(set)
        for symbol, denotations in self.extension.items():
            for denotation in denotations:
                symbols[denotation].add(symbol)
        return symbols

    def denotations(self, 
                snt: Sentence,
                var: Variable,
                g: Dict[Variable,Entity] = dict()) -> set:
        denotation = set()
        for entity in self.entities:
            g_new = copy(g)
            g_new[var] = entity
            if self.eval(snt, g_new):
                denotation.add(entity)
        return denotation

    def projection(self, snt: Sentence, var: Variable) -> DomainModel:
        """implement domain model projection"""
        pass

    def eval(self,
            snt: Sentence,
            g: Dict[Variable,Entity] = dict()) -> bool:
        """Evaluate snt with respect to given assignment function g"""

        if isinstance(snt, AtomicSentence):        

            name, terms = snt.name, snt.terms
            # handle unseen 
            if name not in self.extension.keys():
                return False
            if len(terms) == 1 :
                t = terms[0]
                ground = g[t] if not isinstance(t,int) else t
            else:
                ground = tuple([g[t] if not isinstance(t,int) else t for t in terms])

            return ground in self.extension[name]

        elif isinstance(snt,AndSentence):
            return (self.eval(snt.left,g) and self.eval(snt.right,g))
        elif isinstance(snt,OrSentence):
            return (self.eval(snt.left,g) or self.eval(snt.right,g))
        elif isinstance(snt,ImpSentence):
            return (not self.eval(snt.left,g) or self.eval(snt.right,g))
        elif isinstance(snt,IffSentence):
            return (self.eval(snt.left,g) == self.eval(snt.right,g))

        elif isinstance(snt, NegatedSentence):
            return not self.eval(snt.snt, g)
        elif isinstance(snt, QuantifierSentence):
            name, var, rstr, body = snt.name, snt.var, snt.rstr, snt.body
            nums = get_nums(name)
            rstr_eval = self.denotations(rstr, var, g)
            body_eval = self.denotations(body, var, g)

            rstr_body_size = len(rstr_eval.intersection(body_eval))
            rstr_size = len(rstr_eval)
            
            if self.grm.A.matches(name):
                return rstr_body_size >= 1
            elif self.grm.EVERY.matches(name):
                return rstr_body_size == rstr_size
            elif self.grm.BOTH.matches(name):
                return rstr_body_size == 2 and rstr_size == 2
            elif self.grm.EXACTLY.matches(name):
                return rstr_body_size == nums[0]
            elif self.grm.AT_MOST.matches(name):
                return rstr_body_size <= nums[0]
            elif self.grm.AT_LEAST.matches(name):
                return rstr_body_size >= nums[0]
            elif self.grm.UNIQUE.matches(name):
                return rstr_body_size == nums[0] and rstr_size == nums[0]
            elif self.grm.ALL_BUT.matches(name):
                return rstr_body_size == (rstr_size - nums[0]) and rstr_size >= nums[0]
            elif self.grm.OF_THE.matches(name):
                return rstr_body_size == nums[0] and rstr_size == nums[1]


    def referent(self, refexp: RefExp) -> frozenset:
        """Compute the referent of a referential expression
        :param refexp: referential expression
        
        :return referent (frozenset)"""

        name, var, snt = refexp.name, refexp.var, refexp.snt
        nums = get_nums(name)

        domain = list(self.denotations(snt,var, g={}))
        size = len(domain)
        U = reduce(lambda u, den: u + [ss + [den] for ss in u], domain,[[]])
        #remove empty list & enable set of sets
        U = [frozenset(u) for u in U if u != []]
        if self.grm.A.matches(name):
            return frozenset(u for u in U if len(u) == 1)
        elif self.grm.EVERY.matches(name):
            return frozenset(u for u in U if len(u) == size)
        elif self.grm.EXACTLY.matches(name):
            return frozenset(u for u in U if len(u) == nums[0])
        elif self.grm.AT_MOST.matches(name):
            return frozenset(u for u in U if len(u) <= nums[0])           
        elif self.grm.AT_LEAST.matches(name):
            return frozenset(u for u in U if len(u) >= nums[0])
        elif self.grm.UNIQUE.matches(name):
            return frozenset(u for u in U if len(u) == nums[0]) if size == nums[0] else frozenset()
        elif self.grm.BOTH.matches(name):
            return frozenset(u for u in U if len(u) == 2) if size == 2 else frozenset()
        elif self.grm.ALL_BUT.matches(name):
            return frozenset(u for u in U if len(u) == nums[0]) if size >= nums[0] else frozenset()
        elif self.grm.OF_THE.matches(name):
            return frozenset(u for u in U if len(u) == nums[0]) if size == nums[1] else frozenset()


def merge_domain_models(fst: DomainModel, snd: DomainModel) -> DomainModel:
    """Merge two domain models with same entities and non-overaling extensions
    :param fst: first domain model
    :param snd: second domain model

    :return merged domain model
    """
    assert fst.entities == snd.entities
    assert fst.entity_symbol == snd.entity_symbol
    assert (set(fst.extension.keys()) & set(snd.extension.keys())) == set([fst.entity_symbol])

    entities  = fst.entities
    extension = defaultdict(set, {**fst.extension, **snd.extension})

    return DomainModel(entities, extension)