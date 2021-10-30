from typing import  List, Tuple, Dict
import re, subprocess, os, multiprocessing
from itertools import  product
from functools import reduce

from sympy import Symbol as sympy_symbol
from sympy.logic.boolalg import to_cnf 

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
        valuation = self.model.extension
        return atom in valuation.keys() and terms in valuation[name]

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
        self.add_sentence(snt)

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
        
        return float(wmc)

    def _estimate_atom(self,
                    atom: AtomicSentence) -> Tuple[AtomicSentence,float]:
        """Estimate atom
        :param atom: atomic sentence """
        if self.atom_in_model(atom):
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


class HeadOnlyReasoner(Reasoner):
    
    def _filter(self, snt: Sentence) -> List[Sentence]:
        if isinstance(snt, AtomicSentence):
            return [snt]
        elif isinstance(snt, AndSentence):
            return self._filter(snt.left) + self._filter(snt.right)
        elif isinstance(snt, QuantifierSentence):
            return []

    def add_refexp(self, refexp: RefExp, model: DomainModel) -> None:
        snt = reduce(lambda x,y: AndSentence(x,y), self._filter(refexp.snt))

        refexp = RefExp([refexp.name, refexp.var, str(snt)])

        return super().add_refexp(refexp, model)


class ExistentialReasoner(Reasoner):

    def _filter(self, snt):
        if isinstance(snt, AtomicSentence):
            return snt
        elif isinstance(snt, NegatedSentence):
            return NegatedSentence(snt=self._filter(snt.snt))
        elif isinstance(snt, ConnectiveSentence):
            return type(snt)(left=self._filter(snt.left), 
                            right=self._filter(snt.right))
        elif isinstance(snt, QuantifierSentence):
            return QuantifierSentence(name = '_a_q',
                                var=snt.var,
                                rstr=self._filter(snt.rstr),
                                body=self._filter(snt.body)) 

    def add_refexp(self, refexp: RefExp, model: DomainModel) -> None:

        refexp = RefExp(['_a_q', refexp.var, str(self._filter(refexp.snt))])

        return super().add_refexp(refexp, model)

