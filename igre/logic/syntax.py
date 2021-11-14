"""
Syntax support for predicate logic with generalized quantifiers
"""
from copy import deepcopy
from typing import List, Dict, Union
import re
from functools import reduce
from dataclasses import  dataclass
from pyparsing import (Word, Regex, Suppress, oneOf, alphas, Forward, Optional,
                Group, infixNotation,delimitedList, opAssoc, Keyword, nums, ParseResults)

class Grammar:
    """Grammar elements for predicate logic with generalized quantifiers"""
    ## Punctuation 
    LPAREN = Suppress("(")
    RPAREN = Suppress(")")
    LANGLE = Suppress("<")
    RANGLE = Suppress(">")
    LBRACKET = Keyword("[")
    RBRACKET = Keyword("]")
    COMMA = Suppress(',')
    DOT = Suppress('.')
    ## Logical Connectives
    NEGATION = oneOf("! ¬ neg ~")
    OR = oneOf("v or |")
    AND = oneOf("^ and &")
    IMPLIES = oneOf("=> ->")
    BICONDITION = oneOf("<=> <->")
    BICONNECTIVE = OR|AND|IMPLIES|BICONDITION
    CONNECTIVE = NEGATION|BICONNECTIVE
    # Quantifiers
    A = Keyword("_a_q")
    EVERY = Regex("_every_q|_all_q")
    BOTH =  Keyword("_both_2_q")
    EXACTLY = Regex(r"_exactly_\d+_q")
    AT_MOST = Regex(r"_at_most_\d+_q")
    AT_LEAST = Regex(r"_at_least_\d+_q")
    UNIQUE = Regex(r"_the_\d+_q")
    ALL_BUT = Regex(r"_all_but_\d+_q")
    OF_THE = Regex(r"_\d+_of_the_\d+_q")
    # Qauntifier agregated
    NUM = EXACTLY|AT_MOST|AT_LEAST|UNIQUE|ALL_BUT|OF_THE
    QUANT = A|EVERY|BOTH|NUM
    PRESUPOSITION = EVERY|BOTH|UNIQUE|ALL_BUT
    ## atom elements 
    VAR = Regex(r'[x,e]\d+')
    CONSTANT = Regex(r'\d+')
    TERM = VAR|CONSTANT
    PRED = Word(alphas + "_" + nums + "/")

def get_nums(token: str) -> List[int]:
    """get numbers in the token e.g. numeric quantifier"""
    return [int(num) for num in re.findall(r'\d+',token)]


class NotSupportedSentenceError(Exception):
    def __init__(self, snt: str) -> None:
        super(NotSupportedSentenceError, self).__init__(snt)


class NotSupportedTokenError(Exception):
    def __init__(self, token: str) -> None:
        super(NotSupportedTokenError, self).__init__(token)


Variable = str
Symbol = str

@dataclass
class Sentence:
    """sentence factory class"""
    def __init__(self, tokens: List[str]) -> None:
        self.snt = make_sentence(tokens[0])

    @property
    def symbols(self) -> set:
        """symbol factory"""
        raise NotImplementedError
    
    @property
    def symbols_all(self) -> list:
        """all symbols factory"""
        raise NotImplementedError

    def __str__(self) -> str:
        return str(self.snt)
        
    def __repr__(self) -> str:
        return str(self)


class AtomicSentence(Sentence):
    """atomic sentences of the form pred(term_1,term_2,...term_n)"""
    def __init__(self, name: str, terms: List[str]) -> None:
        self.name = name
        self.arity = len(terms)            
        self.terms = terms
                    
    def assign(self, g: Dict)-> None:
        """assign variables to specific value"""
        self.terms = [t if t not in g.keys() else g[t] for t in self.terms]

    @property
    def symbols(self) -> set:
        return set([(self.name, self.arity)])

    @property
    def symbols_all(self) -> list:
        return [self.name]

    def __str__(self) -> str:
        args = ",".join([str(t) for t in self.terms])
        return f"{self.name}({args})"

    def __eq__(self, other: Sentence) -> bool:
        return str(self) == str(other)

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

class NegatedSentence(Sentence):
    """negated sentences of the form ¬sentence"""
    def __init__(self, snt: Sentence) -> None:
        self.snt = snt

    def assign(self, g: dict)-> None:
        self.snt.assign(g)

    @property
    def symbols(self) -> set:
        return self.snt.symbols
    @property
    def symbols_all(self) -> list:
        return self.snt.symbols_all

    def __str__(self) -> str:
        return f"!({self.snt})"

    def __eq__(self, other: Sentence) -> bool:
        eq_type = isinstance(other, NegatedSentence)
        eq_snt = self.snt == other.snt
        return all([eq_type,eq_snt])

class ConnectiveSentence(Sentence):
    """connective sentence of the form sentence conn sentence"""
    def __init__(self,
                left: Sentence,
                conn: str,
                right: Sentence) -> None:
        self.left = left
        self.conn = conn
        self.right = right

    def assign(self, g:dict)-> None:
        self.left.assign(g)
        self.right.assign(g)

    @property
    def symbols(self) -> set:
        return self.left.symbols | self.right.symbols

    @property
    def symbols_all(self) -> list:
        return self.left.symbols_all + self.right.symbols_all
    

    def __str__(self) -> str:
        return f"{self.left} {self.conn} {self.right}"
    
    def __eq__(self, other: Sentence) -> bool:
        eq_type = isinstance(other, ConnectiveSentence)
        eq_conn = self.conn == other.conn
        eq_left = self.left == other.left
        eq_right =  self.right == other.right
        return  all([eq_type, eq_conn, eq_left, eq_right])

class AndSentence(ConnectiveSentence):
    def __init__(self,
                left: Sentence,
                right: Sentence) -> None:
        super().__init__(left,"^",right)


class OrSentence(ConnectiveSentence):
    def __init__(self,
                left: Sentence,
                right: Sentence) -> None:
        super().__init__(left,"v",right)


class ImpSentence(ConnectiveSentence):
    def __init__(self,
                left: Sentence,
                right: Sentence) -> None:
        super().__init__(left,"=>",right)


class IffSentence(ConnectiveSentence):
    def __init__(self,
                left: Sentence,
                right: Sentence) -> None:
        super().__init__(left,"<=>",right)


class QuantifierSentence(Sentence):
    """quantifier sentence of the form quantifer var.(rstr,body)"""
    def __init__(self,
                name: str,
                var: str,
                rstr: Sentence,
                body: Sentence)->None:
        self.name = name
        self.var =  var     
        self.rstr = rstr
        self.body = body

    def assign(self, g:dict)-> None:
        self.body.assign(g)

    @property
    def symbols(self) -> set:
        return self.rstr.symbols | self.body.symbols

    @property
    def symbols_all(self) -> list:
        return self.rstr.symbols_all + self.body.symbols_all

    def __str__(self) -> str:
        return f"{self.name} {self.var}.({self.rstr},{self.body})"

    def __eq__(self,
            other:Sentence)->bool:
        eq_type = isinstance(other, QuantifierSentence)
        eq_name = self.name == other.name
        eq_var = self.var == other.var
        eq_rstr = self.rstr == other.rstr
        eq_body = self.body == other.body

        return  all([eq_type, eq_name,eq_var,eq_rstr,eq_body])


def make_sentence(tokens: List[str]) -> Sentence:
    """lookahead Sentence factory"""
    grm = Grammar()
    if  grm.QUANT.matches(tokens[0]): #len(tokens) == 4 and
        name = tokens[0]
        var = tokens[1]
        rstr = make_sentence(tokens[2])
        body = make_sentence(tokens[3])
        return QuantifierSentence(name,var,rstr,body)
    elif  grm.NEGATION.matches(tokens[0]): # len(tokens) == 2 and
        snt  = make_sentence(tokens[1])
        return NegatedSentence(snt)
    elif  grm.BICONNECTIVE.matches(tokens[1]): #len(tokens) == 3 and
        left = make_sentence(tokens[0])
        conn = tokens[1]
        # handle left assoativity
        if len(tokens) == 3:
            right = make_sentence(tokens[2])
        else:
            right = make_sentence(ParseResults(tokens[2:])) 
        if grm.AND.matches(conn):
            return AndSentence(left,right)
        elif grm.OR.matches(conn):
            return OrSentence(left,right)
        elif grm.IMPLIES.matches(conn):
            return ImpSentence(left,right)
        elif grm.BICONDITION.matches(conn):
            return IffSentence(left,right)
        else:
            raise NotSupportedTokenError(conn)
    elif  grm.PRED.matches(tokens[0]): #len(tokens) >=1 and
        name = tokens[0]
        terms = tokens[1:]
        return AtomicSentence(name,terms)
    else:
        raise NotSupportedSentenceError(tokens)

class RefExp:
    def __init__(self, tokens: List[str]) -> None:
        super().__init__()
        self.name = tokens[0]
        self.var =  tokens[1]        
        self.snt = make_sentence(tokens[2])

    def __str__(self) -> str:
        return f"<{self.name} {self.var}.{self.snt} >"

    def to_snt(self, entities: List[str]) -> Sentence:
        snts = []
        for entity in entities:
            snt = deepcopy(self.snt)
            snt.assign({self.var: entity})
            snts.append(snt)

        print(f"snsnts: {snts}")
        if len(snts) == 1:
            return snts[0]
        else:
            return reduce(lambda x,y: AndSentence(x,y), snts)
    
    @property
    def symbols(self) -> set:
        """ Get predicates part of a sentence """
        return self.snt.symbols

class LogicSentenceParser:

    def __init__(self) -> None:
        """parser for predicate logic with generalized quantifiers"""
        super().__init__()
        # Grammar for Predicate Logic with Generalized Quantifiers
        grm = Grammar()
        expr = Forward()
        term  = Group(grm.PRED + grm.LPAREN + delimitedList(grm.TERM) + grm.RPAREN)
        quant = Group(grm.QUANT + grm.VAR + grm.DOT + grm.LPAREN + expr + grm.COMMA + expr + grm.RPAREN)

        operands = quant | term
        expr << infixNotation(operands,[
                            (grm.NEGATION, 1, opAssoc.RIGHT),
                            (grm.AND, 2, opAssoc.LEFT),
                            (grm.OR, 2, opAssoc.LEFT),
                            (grm.IMPLIES, 2, opAssoc.LEFT),
                            (grm.BICONDITION, 2, opAssoc.LEFT),
                            ])
        # Optional dot allows to stop recursion at top level
        expr = (expr+Optional(grm.DOT)).setParseAction(Sentence)

        self.parser = expr


    def __call__(self, str_expr: str) -> Sentence:
        return self.parser.parseString(str_expr)[0].snt


class LogicRefExpParser:

    def __init__(self) -> None:
        super().__init__()
        """parser for logical forms of referential expressions"""
        # Grammar for Predicate Logic with Generalized Quantifiers
        grm = Grammar()
        expr = Forward()
        term  = Group(grm.PRED + grm.LPAREN + delimitedList(grm.TERM) + grm.RPAREN)
        quant = Group(grm.QUANT + grm.VAR + grm.DOT + grm.LPAREN + expr + grm.COMMA + expr + grm.RPAREN)

        operands =  quant | term 
        expr << infixNotation(operands,[
                            (grm.NEGATION, 1, opAssoc.RIGHT),
                            (grm.AND, 2, opAssoc.LEFT),
                            (grm.OR, 2, opAssoc.LEFT),
                            (grm.IMPLIES, 2, opAssoc.LEFT),
                            (grm.BICONDITION, 2, opAssoc.LEFT),
                            ])
        # Grammar for logical forms of referential expressions
        expr = (grm.LANGLE+grm.QUANT+grm.VAR+grm.DOT+expr+grm.RANGLE).setParseAction(RefExp)

        self.parser = expr

    def __call__(self, str_expr: str) -> RefExp:
        return self.parser.parseString(str_expr)[0]
