from typing import List, Tuple, Iterator
import argparse, re, subprocess, os
from copy import deepcopy

from delphin.ace import ACEParser
from delphin.mrs import EP, HCons, MRS
from delphin.lnk import Lnk
from delphin.codecs import mrsprolog

class Mudger:
    """
    Elemetary predication (EP) in MRS mudger (EP combiner) to abstract 
    away from too detail ERG analysis. 
    
    New mudge rules are introduced as static methods to this class starting
    with `mudge`.

    """
    def __call__(self, mrs:MRS, surface: str) -> MRS:
        """dynamicly apply mudge rules until it is not possible anymore"""
        rules = [i for i in dir(self.__class__) if i.startswith('mudge')]
        for rule in rules:
            mrs = getattr(self,rule)(mrs, surface)    
        return mrs
    
    @staticmethod
    def update_eps(mrs: MRS, 
                remove_eps: List[EP], 
                add_eps: List[Tuple[EP,int]]) -> MRS:
        """update the list of EPs in MRS
        :param remove_eps: list of EPs to remove from MRS
        :param add_eps: list of tuples of eps and insert locations
        
        :return update MRS structure"""
        mrs.predications = list(filter(lambda x: x not in remove_eps, 
                                                    mrs.predications))
        
        for ep, idx in add_eps:
            mrs.predications.insert(idx,ep) 
        
        return mrs 
        
    @staticmethod
    def get_eps(eps: List[EP],
                surface:str,
                pattern:str) -> Iterator[Tuple[List[EP],Lnk]]:
        """Iterator of EPs to mudge based on the specific pattern"""

        for match in re.finditer(pattern, surface):

            start, end = match.start(), match.end()
            span = lambda ep: ep.cfrom >= start and ep.cto <= end
            lnk = Lnk(f"<{start}:{end}>")
            filtered = list(filter(span, eps))

            yield filtered, lnk

    @staticmethod
    def mudge__2_gen_q(mrs: MRS,
                    surface: str) -> MRS:
        """Mudge generalized quantifiers
        :param mrs: instance of MRS object 
        :param surface: string representation of the surface format
        
        :return updated MRS object"""
        gen_q_patterns = [
            (r'all but \w+',"all_but"),
            (r'the \w+', "the"),
            (r'exactly \w+',"exactly"),
            (r'both', "both"),
            (r'at least \w+', "at_least"),
            (r'at most \w+', "at_most"),
        ]
        for pattern, str_pattern in gen_q_patterns:
            add_list, remove_list = [],[]
            for eps, lnk in Mudger.get_eps(mrs.predications,
                                        surface,
                                        pattern):
                if len(eps) == 0:
                    continue 
                idx = mrs.predications.index(eps[0])
                q, card = eps[len(eps)-2:]
                ep = EP(predicate=f"_{str_pattern}_{card.args['CARG']}_q", 
                        label=q.label, 
                        args={'ARG0': q.args['ARG0'], 
                        'RSTR': q.args['RSTR'], 
                        'BODY': q.args['BODY']},
                        lnk=lnk)
                add_list.append((ep,idx))
                remove_list += eps

            mrs = Mudger.update_eps(mrs, remove_list, add_list)

        return mrs

    @staticmethod
    def mudge__1_to_the_x_of(mrs: MRS,
                        surface: str,
                        pattern :str =r'to the (left|right) of') -> MRS:
        """mudged EPS from constuction `to the x of` to just `x`
        :param mrs: instance of MRS object 
        :param surface: string representation of the surface format
        :param pattern: pattern to mudge
        
        :return updated MRS object"""
        add_list, remove_list = [],[]
        for eps, lnk in Mudger.get_eps(mrs.predications,
                                    surface,
                                    pattern):
            assert len(eps) == 3 
            idx = mrs.predications.index(eps[0])
            _to, _the, _x, = eps
            ep = EP(predicate=_x.predicate, 
                        label=_to.label, 
                        args={'AR0': _to.args['ARG1'],
                            'ARG1': _x.args['ARG1']},
                        lnk=lnk)
            add_list.append((ep, idx))
            remove_list += eps
            
            ## remove hcon
            mrs.hcons.remove(HCons(_the.args['RSTR'],'qeq', _x.label))

        return Mudger.update_eps(mrs, remove_list, add_list)

    @staticmethod
    def mudge__1_n_of_the_m(mrs: MRS,
                        surface: str,
                        pattern: str = r'\w+(?<!right)(?<!left) of the \w+') -> MRS:
        """mudged EPS from constuction `n of the m` to single EP
        :param mrs: instance of MRS object 
        :param surface: string representation of the surface format
        :param pattern: pattern to mudge
        
        :return updated MRS object"""

        
        add_list, remove_list = [],[]
        for eps, lnk in Mudger.get_eps(mrs.predications,
                                    surface,
                                    pattern):
            assert len(eps) == 5 
            idx = mrs.predications.index(eps[0])
            _part_of ,_udef_q, n, _the, m = eps

            ep = EP(predicate=f"_{n.args['CARG']}_of_the_{m.args['CARG']}_q", 
                    label=_the.label, 
                    args={'ARG0': _the.args['ARG0'], 
                        'RSTR': _the.args['RSTR'], 
                        'BODY': _the.args['BODY']},
                    lnk=lnk)
            


            add_list.append((ep,idx))
            remove_list += eps
            # predicate just before above needs to change its argument 
            for arg_name, arg_val in mrs.predications[idx-1].args.items():
                if arg_val ==  _part_of.args['ARG0']:
                    mrs.predications[idx-1].args[arg_name] = _part_of.args['ARG1']

            # remove hcon
            mrs.hcons.remove(HCons(_udef_q.args['RSTR'],'qeq', n.label))


        return Mudger.update_eps(mrs, remove_list, add_list)
    

class RefExpParser:

    def __init__(self, 
                grm_path: str = './external/erg-2018-x86-64-0.9.31.dat',
                ace_path: str = './external/ace',
                utool_path: str = './external/Utool-3.1.1.jar',
                tmp_path: str = './tmp.mrs.pl',
                ) -> None:
        """
        Referential expression parser to its logical form.

        :param grm_path: path to the grammar file
        :param ace_path: path to the ACE binaries
        :param utool_path: path to Utool jar-file
        """
        self.grm_path = grm_path
        self.ace_path = ace_path
        self.utool_path = utool_path
        self.tmp_path = tmp_path
        self.mudger = Mudger()

    def remove_underspecification(self, 
                                mrs: MRS,
                                use:List[str] = ['x']) -> str:
        """run Utool and remove underspecitication from the MRS representation
        :param mrs: MRS object instance
        :param use: List of variables to use 
        """
        def clean_predicate(s):
            "clean predicate from extra annocations given by ERG"
            s = s.group(0)
            if s[len(s)-1] == 'q':
                return s
            else:
                 return re.search(r"[a-z]+", s).group()
        SEP="<SPACE>"
        transforms = [
            (r"_\w+\/?\w*", clean_predicate),
            (f"{SEP}", " "),
            (r"\'",""),
            (r"\[",""),
            (r"\]",""),
            (r"&"," ^ ")
        ]

        # Add terms for predicate string representation
        for p in mrs.predications:
            if p.is_quantifier():
                p.predicate +=  f"{SEP}{p.iv}."
            elif 'neg' == p.predicate:
                continue
            else:
                # removes unmodeled terms like e and i
                vs = p.args.values()
                terms = ",".join([ v for v in vs if any([u in v for u in use])])
                args = f"({terms})"
                p.predicate += args

        with open(self.tmp_path, 'w+') as f:
            f.write(mrsprolog.encode(mrs))

        parse = subprocess.run(['java', 
                                '-jar', 
                                self.utool_path, 
                                'solve', 
                                '-I', 
                                'mrs-prolog', 
                                '-O', 
                                'term-prolog', 
                                self.tmp_path], 
                                capture_output=True).stdout.\
                                decode('UTF-8').splitlines()
        # os.remove(self.tmp_path)

        q_order = "\S*".join([p.predicate for p in 
                mrs.predications if p.is_quantifier() or p.predicate == 'neg'])
                
        parse = [p for p in parse if re.search(q_order, p)][0]

        ## Post-processing
        parse = parse[:len(parse)-1] # remove training comma
        for in_pred,out_pred in transforms:
            parse = re.sub(in_pred,out_pred, parse)
    
        return parse

    @staticmethod
    def to_ref_exp(lf:str ) -> str:
        """Convert logical form to match syntax of referential expressions"""
        match = re.match(r"_\w*_q x\d+.", lf)
        quant, var = (lf[match.start():match.end()]).split()
        unk_token = f",unknown{(var)})"
        ref_exp = lf[match.end()+1:len(lf)-1-len(unk_token)]
        ref_exp = f"<{quant} {var} {ref_exp}>"
        return ref_exp
    
    def parse(self, surface: str, max_parse: int = 5) -> str:
        """parse referential expression
        :param surface: surface form of a referential expression
        :param max_parse: number of parses to explore
        
        :return str of a logical form of referential expressions."""

        # Parse referential expression to MRS 
        with ACEParser(grm=self.grm_path, executable=self.ace_path) as parser:
            response = parser.interact(surface)
        # ACE returns N most probable results by MaxEnt model, select 0th
        mrs = response.result(0).mrs()
        # Mudge predications for application-specific analysis
        mrs = self.mudger(mrs, surface)
        # Remove underspecification and convert MRS to first-order logic
        lf = self.remove_underspecification(mrs)
        # Convert first-order logic formula to the logical form of RefExp
        ref_exp = RefExpParser.to_ref_exp(lf)

        return ref_exp

            
    def __call__(self, surface:str) -> str:
        """Parse referential expression to of its logical form string"""

        ## handle logical connectives 
        if exp := re.search(r' and | or ',surface):
            fst_ref_exp = surface[:exp.start()]
            fst = self(fst_ref_exp)
            conn = ' ^ ' if exp.group(0) == ' and ' else ' v '

            snd_ref_exp = surface[exp.end():]
            snd = self(snd_ref_exp)
            return " ".join([fst, conn, snd])
        # handle negation at the begining of RefExp
        elif exp := re.search(r'^not ', surface):
            ref_exp = surface[exp.end():]
            r = self(ref_exp)
            return r[:r.index(".")+1] + 'neg(' + r[r.index(".")+1:-1] + ')>'
        else:
            return self.parse(surface)

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='grammar based parser for referential \
                        expression to their logical form')
    parser.add_argument('--grm',
                    default='./external/erg-2018-x86-64-0.9.31.dat',
                    help='Path to grammar')
    parser.add_argument('--ace',
                    default='./external/ace',
                    help='Path to ACE')
    parser.add_argument('--utool',
                    default='./external/Utool-3.1.1.jar',
                    help='Path to Utool jar file')
    parser.add_argument('--refexp',
                    default='a circle behind the one square.',
                    help='referential expression to parse')
    args = parser.parse_args()

    refexp_parser = RefExpParser(args.grm, args.ace, args.utool)

    refexp = refexp_parser(args.refexp)

    print(refexp)
    