from typing import List
import re
from copy import deepcopy 
from itertools import  combinations,chain
from functools import reduce

from .syntax import (Grammar, ImpSentence, RefExp, Sentence, AtomicSentence, NegatedSentence, 
            AndSentence, OrSentence, ConnectiveSentence,
            QuantifierSentence, get_nums, NotSupportedTokenError)

def refexp2sentence(refexp:RefExp, 
                    referents:frozenset,
                    domain:set) -> Sentence:
    """Construct a sentence based on ref_exp and its referent
    :param refexp: referential expression 
    :param referents: frozenset of frozensets
    :param domain: set of entities in the universe of discourse """
    grm = Grammar()
    print(f"refexp to reduce: {refexp}")
    name, var, snt = refexp.name, refexp.var, refexp.snt
    refs = [elem for referent in referents for elem in referent]
    
    snts, nsnts = [], []
    for individual in domain:
        snt_new = deepcopy(snt)
        snt_new.assign({var:individual})
        if individual in refs:
            snts.append(snt_new)
        else:
            nsnts.append(NegatedSentence(snt_new))

    # if grm.OF_THE.matches(name):
    #     """special treatment due to disjunctive nature of sentence formation"""
    #     n,m = [int(i) for i in re.findall(r'\d+', name)]
    #     snts = [reduce(lambda x,y: AndSentence(x,y),chain([snts,list(combinations(nsnts, m-n))]))]
    #     return reduce(lambda x,y: OrSentence(x,y), )

    if grm.PRESUPOSITION.matches(name) or grm.OF_THE.matches(name):
        return reduce(lambda x,y: AndSentence(x,y), snts+nsnts)
    else:
        return reduce(lambda x,y: AndSentence(x,y), snts)
    

def propositionalize(snt: Sentence, domain: set)-> Sentence:
    """ Propositionalization

    convert predicate logic sentence to propositional logic sentence by
    removing quantifiers

    :param snt: predicate logic sentence
    :param domain: domain of discouse

    :return propositional logic sentence (without quantifiers)
    """
    grm = Grammar()

    def choose_n(domain, n):
        """choose n elements from the domain, return set"""
        for elem in combinations(domain, n):
            yield set(elem)

    def factory(rstr, body, var, val, polarity, conn):
        """factory for (rstr conn body) with var assigned to val"""
        var = str(var)
        left, right = deepcopy(rstr), deepcopy(body)
        left.assign({var:val})
        right.assign({var:val})
        snt = conn(left,right)

        return snt if polarity else NegatedSentence(snt)

    def conj(domain: set, snts: List[Sentence]) -> AndSentence:
        """conjunction of sentences"""
        return reduce(lambda x,y: AndSentence(propositionalize(x,domain),
                                            propositionalize(y,domain)),snts)

    def disj(domain: set, snts: List[Sentence]) -> OrSentence:
        """disjunction of sentences"""
        return reduce(lambda x,y: OrSentence(propositionalize(x,domain),
                                            propositionalize(y,domain)),snts)

    def situation(rstr,body,var,pos,neg,conn=AndSentence):
        """particular situation as conjunction of rstr and body assigned"""
        pos_snts = [factory(rstr,body,var,val,True,conn) for val in pos]
        neg_snts = [factory(rstr,body,var,val,False,conn) for val in neg]
        return conj(domain, pos_snts + neg_snts)

    if isinstance(snt, AtomicSentence):
        # assure no variables left
        assert all([isinstance(term, int) for term in snt.terms])
        return snt
    elif isinstance(snt, ConnectiveSentence):
        left = propositionalize(snt.left, domain)
        right = propositionalize(snt.right, domain)
        return type(snt)(left,right)
    elif isinstance(snt, NegatedSentence):
        return type(snt)(propositionalize(snt.snt, domain))
    elif isinstance(snt, QuantifierSentence):

        name, var, rstr, body = snt.name, snt.var, snt.rstr, snt.body
        nums = get_nums(name)

        # limit search by only consider elements that are known from restrictor
        # pes = model.proj(rstr, var).entities
        if grm.A.matches(name):
            # _a_q x .(rstr(x), body(x)):
            # v_{u in U} [rstr(u) ^ body(u)] 
            snts = [factory(rstr,body,var,val,True,AndSentence) for val in domain]
            return disj(domain, snts)
        elif grm.EVERY.matches(name):
            # _every_q x .(rstr(x), body(x)):
            # ^_{u in U} [rstr(u) => body(u)] 
            snts = [factory(rstr,body,var,val,True,ImpSentence) for val in domain]
            return conj(domain, snts)
        elif grm.EXACTLY.matches(name):
            # _exactly_n x. (rstr(x), body(x)):
            # v (for n: is true)
            assert len(nums) == 1
            n = nums[0]
            snts = [situation(rstr,body,var,e,{}) for e in choose_n(domain,n)]
            return disj(domain,snts)
        elif grm.AT_MOST.matches(name):
            assert len(nums) == 1
            n = nums[0]
            snts = [situation(rstr,body,var,e,{}) for e in choose_n(domain,n)]
            return disj(domain,snts)
        elif grm.AT_LEAST.matches(name):
            assert len(nums) == 1
            n = nums[0]
            snts = [situation(rstr,body,var,e,{}) for e in choose_n(domain,n)]
            return disj(domain,snts)
        elif grm.BOTH.matches(name):
            n = 2
            snts = [situation(rstr,body,var,e,domain-e) for e in choose_n(domain,n)]
            return disj(domain,snts)
        elif grm.UNIQUE.matches(name):
            assert len(nums) == 1
            n = nums[0]
            snts = [situation(rstr,body,var,e,domain-e) for e in choose_n(domain,n)]
            return disj(domain,snts)
        elif grm.ALL_BUT.matches(name):
            assert len(nums) == 1
            n = nums[0]
            snts = [situation(rstr,body,var,set(e),domain-e, ImpSentence) for e in choose_n(domain,n)]
            return disj(domain,snts)
        elif grm.OF_THE.matches(name):
            assert len(nums) == 2
            n, m = nums
            snts = [situation(rstr,body,var,e,domain-e) for e in choose_n(domain,n)]
            print(snts)
            return disj(domain,snts)
        else:
            raise NotSupportedTokenError(name)