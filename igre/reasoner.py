from typing import  List, Tuple, Dict
import re, subprocess, os, multiprocessing
from copy import deepcopy
from itertools import  product
from functools import reduce

from sympy import Symbol as sympy_symbol
from sympy.logic.boolalg import to_cnf
from func_timeout import func_timeout, FunctionTimedOut 

from igre.logic import *

class Reasoner:

    def __init__(self,
                model: DomainModel = DomainModel(),
                addmc_path: str = "./external/addmc",
                weight: float = 0.5,
                ) -> None:
        """
        Reasoner to estimate beliefs abou the truth-values of atoms

        :param theory: list of Sentences that have to be true
        :param model: previously known model
        :param addmc_path: path th addmc binaries
        :param weight: weight for atom
        :param use_head_only: if true, ablate the model 
        """
        self.theory: List[Sentence] = []
        self._atoms: List[AtomicSentence] = []
        self._weight: List[float] = []
        self._cnf: str = ""
        self._num_clauses: int = 0
        self.model: DomainModel = model
        self.addmc_path: str = addmc_path
        self.weight: float = weight

        if len(self.model.extension) != 0:
            for symbol, denotations in self.model.extension.items():
                for denotation in denotations:
                    if isinstance(denotation,int):
                        atom = AtomicSentence(symbol,[str(denotation)])
                    else:
                        atom = AtomicSentence(symbol,[str(e) for e in denotation])
                    self.add_sentence(atom)

    def atom_in_model(self, atom: AtomicSentence) -> bool:
        "return true if atom is in the initial model"
        name, terms = atom.name, atom.terms
        if atom.arity == 1:
            terms = terms[0]
        key_in = atom.name in self.model.extension.keys()
        ext_in = terms in self.model.extension[name]
        return key_in and ext_in

    def __get_cnf(self, snt: Sentence) -> Tuple[str,int]:
        """convert sentence to conjunctive normal form (CNF)
        :param snt: sentence to convert

        return: CNF, number of clauses in CNF"""
        def to_sympy(s):
            """convert sentence to sympy formula"""
            if isinstance(s, AtomicSentence):
                return sympy_symbol(str(self._atoms.index(s)+1))
            elif isinstance(s, NegatedSentence):
                return ~to_sympy(s.snt)
            elif isinstance(s, AndSentence):
                return to_sympy(s.left) & to_sympy(s.right)
            elif isinstance(s, OrSentence):
                return to_sympy(s.left) | to_sympy(s.right)
            elif isinstance(s, ImpSentence):
                return to_sympy(s.left) >> to_sympy(s.right)
            elif isinstance(s, IffSentence):
                return (to_sympy(s.left) >> to_sympy(s.right)) & \
                    (to_sympy(s.left) << to_sympy(s.right))
            else:
                raise NotSupportedSentenceError(s)

        cnf = str(to_cnf(to_sympy(snt)))

        num_clauses = cnf.count("&") + 1 

        ## Post-processing to match ADDMC syntax
        # Update negation symbol
        cnf = cnf.replace("~","-")
        # remove brackets and | symbol for disjunction
        cnf = re.sub(r"\(|\| |\)","",cnf)
        # change & symbol 0 + newline
        cnf = cnf.replace("& ", "0\n")
        # add 0 for the last clause
        cnf += " 0" 

        return cnf, num_clauses
    
    def __update_atom_index(self, snt: Sentence) -> None:
        """update atom index and give each of them a weighting"""
        for symbol, arity in snt.symbols:
            for term in product(self.model.entities,repeat=arity):
                atom = AtomicSentence(name=symbol,terms=term)
                weight = 1.0 if self.model.eval(atom) else self.weight
                # atoms_weights.add((atom, weight))
                if atom not in self._atoms:
                    self._atoms.append(atom)
                    self._weight.append(weight)

    def add_sentence(self, snt: Sentence) -> None:
        """Add grounded sentence to logic theory and update: atom list,
        number of clauses, and cnf """
        print(f'sentence to add: {snt}')
        self.theory.append(snt)
        self.__update_atom_index(snt)
        cnf, num_clauses = self.__get_cnf(snt)
        self._cnf = self._cnf + "\n" + cnf
        self._num_clauses += num_clauses


    def add_refexp(self, refexp: RefExp, model: DomainModel) -> None:
        """Add referential expression and its referent to logic theory"""
        snt = refexp2sentence(refexp, model.referent(refexp), model.entities)
        # propositionalize a sentence 
        snt = propositionalize(snt, model.entities)
        # add sentence to logic theory
        try:
            func_timeout(600, self.add_sentence, args=(snt,))
        except FunctionTimedOut:
            return ""

    def WMC(self,
            query: str = None,
            tmp_path: str = "./tmp.cnf") -> float:
        """Compute weighted model count """

        num_clauses = self._num_clauses + 1 if query else self._num_clauses

        with open(tmp_path, "w+") as file:
            # WMC specification
            file.write(f"p cnf {len(self._atoms)} {num_clauses}\n")
            # enumerate atoms
            for atom_idx, weight in enumerate(self._weight, start=1):
                file.write(f"w {atom_idx} {weight} \n")
            # CNF
            file.write(self._cnf)
            # query atom
            if query:
                file.write(f"\n{self._atoms.index(query)+1} 0")


        wmc = subprocess.run([self.addmc_path, 
                            "--cf",
                            tmp_path,
                            "--wf",
                            "3"],
                            capture_output=True).stdout.decode('UTF-8')

        os.remove(tmp_path)

        # sometimes we get errror message 
        try:
            wmc = float(wmc)
        except ValueError:
            wmc = 0.5
        
        return wmc

    def _estimate_atom(self,
                    atom: AtomicSentence) -> Tuple[AtomicSentence,float]:
        """Estimate atom
        :param atom: atomic sentence """
        if not self.atom_in_model(atom):
            return (atom,self.WMC(query=atom, tmp_path=f'./{atom}.cnf')/self.Z)
        else:
            return (atom, 1.0)


    def estimate(self) -> Dict[AtomicSentence,float]:
        """Estimates truth-values for atoms
        
        :return atom:truth-value estimate dictionary"""

        if len(self.theory) == 0:
            return dict()

        self.Z = self.WMC()

        assert self.Z, "zero partition function"

        # list comprehension is faster than dict comprehension
        # see https://stackoverflow.com/questions/52542742/why-is-this-loop-faster-than-a-dictionary-comprehension-for-creating-a-dictionar
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        return dict(pool.map(self._estimate_atom, self._atoms))

    
    def add_full(self, refexp: RefExp, model: DomainModel) -> None:
        """add full refexp with backoof strategy in case of long computation"""
        rez = self.add_refexp(refexp, model)
        if isinstance(rez, str):
            self.add_head(refexp, model)

    def add_head(self, refexp: RefExp, model: DomainModel) -> None:

        """add referential expression, but only head"""

        def filter(snt: Sentence, var: Variable) -> List[Sentence]:
            if isinstance(snt, AtomicSentence):
                if len(snt.terms) == 1 and snt.terms[0] == var:
                    return [snt]
                else:
                    return []
            elif isinstance(snt, NegatedSentence):
                return filter(snt.snt, var)
            elif isinstance(snt, ConnectiveSentence):
                return filter(snt.left, var) + filter(snt.right, var)
            elif isinstance(snt, QuantifierSentence):
                return filter(snt.body, var)

        snt = reduce(lambda x,y: AndSentence(x,y), filter(refexp.snt, refexp.var))

        update_refexp = deepcopy(refexp)
        update_refexp.snt = snt
        
        self.add_refexp(update_refexp, model)


    def add_exist(self, refexp: RefExp, model: DomainModel) -> None:
        """add referential expression, but only use existential quantifier"""

        def filter(snt):
            if isinstance(snt, AtomicSentence):
                return snt
            elif isinstance(snt, NegatedSentence):
                return NegatedSentence(filter(snt.snt))
            elif isinstance(snt, ConnectiveSentence):
                return type(snt)(left=filter(snt.left), 
                                right=filter(snt.right))
            elif isinstance(snt, QuantifierSentence):
                return QuantifierSentence(name = '_a_q',
                                    var=snt.var,
                                    rstr=filter(snt.rstr),
                                    body=filter(snt.body)) 


        new_refexp = deepcopy(refexp)

        new_refexp.snt = filter(refexp.snt)
        new_refexp.name = '_a_q'

        rez = self.add_refexp(refexp, model)
        if isinstance(rez, str):
            self.add_head(refexp, model)

