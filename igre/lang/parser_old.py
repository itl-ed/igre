"""
Parsing NL to logical form
"""
import argparse, re, subprocess, os

from delphin.ace import ACEParser
from delphin.mrs import EP, HCons
from delphin.codecs import mrsprolog


from pyparsing import Regex


class SemanticParser:

    def __init__(self, 
                erg_path: str = './external/erg-2018-x86-64-0.9.31.dat',
                ace_path: str = './external/ace',
                utool_path:str = './external/Utool-3.1.1.jar',
                tmp_path: str = './tmp/mrs.pl',
                ) -> None:
        super().__init__()

        self.utool_path = utool_path
        self.tmp_path = tmp_path
        self.NLparser = ACEParser(grm=erg_path,executable=ace_path)
        self.transforms = [
            (r"_square\/?\w*","square"),
            (r'_rectangle\/?\w*','rectangle'),
            (r'_triangle\/?\w*','triangle'),
            (r'_pentagon\/?\w*','pentagon'),
            (r'_cross\/?\w*','cross'),
            (r'_circle\/?\w*','circle'),
            (r'_semicircle\/?\w*','semicircle'), 
            (r'_ellipse\/?\w*','ellipse'),
            (r'_red\/?\w*','red'), 
            (r'_green\/?\w*','green'), 
            (r'_blue\/?\w*','blue'), 
            (r'_yellow\/?\w*','yellow'), 
            (r'_magenta\/?\w*','magenta'), 
            (r'_cyan\/?\w*','cyan'), 
            (r'_grey\/?\w*','grey'),
            (r'_below\/?\w*', 'behind'),
            (r'_above\/?\w*', 'above'),
            (r'_object\/?\w*', 'object'),
            (r'unknown','\'unknown\''),
            (r"\'",""),
            (r"\[",""),
            (r"\]",""),
            (r"&"," ^ ")
        ]

    @staticmethod
    def _remove_eps(lf, eps):
        for ep in eps:
            if ep in lf.predications:
                lf.predications.remove(ep)
        return lf

    def __call__(self, ref_exp):

        exp = re.search(' and | or ',ref_exp)
        if exp:
            fst_ref_exp = ref_exp[:exp.start()]
            fst = self.process_unit_ref_exp(fst_ref_exp)
            conn = ' ^ ' if exp.group(0) == ' and ' else ' v '
            snd_ref_exp = ref_exp[exp.end():]
            snd = self.process_unit_ref_exp(snd_ref_exp)

            return " ".join([fst, conn, snd])
        else:
            return self.process_unit_ref_exp(ref_exp)


    def process_unit_ref_exp(self, ref_exp):
        mrs_lf = self.NLparser.interact(ref_exp).result(0).mrs()

        # merge EPs for MW
        patterns = re.findall('to the \w* of|all but \w*|both|exactly \w*|the \w*|at least \w*|at most \w*| not \w*',ref_exp)
        for pattern in patterns:
            match = re.search(pattern, ref_exp)
            merge = [p for p in mrs_lf.predications if p.cfrom >= match.start() and p.cto <= match.end()]
            loc = mrs_lf.predications.index(merge[0])
            # pattern matching
            if re.match('to the \w* of', pattern):
                ep = EP(predicate=pattern.split()[2], 
                            label=merge[0].label, 
                            args={'AR0': merge[0].args['ARG1'], 
                                'ARG1': merge[2].args['ARG1']},
                            surface=pattern) 
                mrs_lf.predications.insert(loc, ep)
                # remove extra qeq 
                mrs_lf.hcons.remove(HCons(merge[1].args['RSTR'],'qeq', merge[2].label))
                mrs_lf = SemanticParser._remove_eps(mrs_lf, merge)
                    
            elif re.match('all but \w*', pattern):
                ep = EP(predicate=f"_all_but_{merge[len(merge)-1].args['CARG']}_q", 
                            label=merge[len(merge)-2].label, 
                            args={'ARG0': merge[len(merge)-2].args['ARG0'], 
                                'RSTR': merge[len(merge)-2].args['RSTR'], 
                                'BODY': merge[len(merge)-2].args['BODY']},
                            surface=pattern)
                mrs_lf.predications.insert(loc, ep)
                mrs_lf = SemanticParser._remove_eps(mrs_lf, merge)

            elif re.match('both', pattern):
                mrs_lf = SemanticParser._remove_eps(mrs_lf, [merge[1]])

            elif re.match('exactly \w*', pattern):
                ep = EP(predicate=f"_exactly_{merge[2].args['CARG']}_q", 
                            label=merge[1].label, 
                            args={'ARG0': merge[1].args['ARG0'], 
                                'RSTR': merge[1].args['RSTR'], 
                                'BODY': merge[1].args['BODY']},
                            surface=pattern)
                mrs_lf.predications.insert(loc, ep)
                mrs_lf = SemanticParser._remove_eps(mrs_lf, merge)
            
            elif re.match('the \w*', pattern):
                if pattern[4:] in ['one','two','three','four','five','six','seven', 'eight', 'nine']:
                    ep = EP(predicate=f"_the_{merge[1].args['CARG']}_q", 
                                label=merge[0].label, 
                                args={'ARG0': merge[0].args['ARG0'], 
                                    'RSTR': merge[0].args['RSTR'], 
                                    'BODY': merge[0].args['BODY']},
                                surface=pattern)
                    mrs_lf.predications.insert(loc, ep)
                    mrs_lf = SemanticParser._remove_eps(mrs_lf, merge)
            elif re.match('at least \w*', pattern):
                ep = EP(predicate=f"_at_least_{merge[len(merge)-1].args['CARG']}_q", 
                            label=merge[len(merge)-2].label, 
                            args={'ARG0': merge[len(merge)-2].args['ARG0'], 
                                'RSTR': merge[len(merge)-2].args['RSTR'], 
                                'BODY': merge[len(merge)-2].args['BODY']},
                            surface=pattern)
                mrs_lf.predications.insert(loc, ep)
                mrs_lf = SemanticParser._remove_eps(mrs_lf, merge)
            elif re.match('at most \w*', pattern):
                ep = EP(predicate=f"_at_most_{merge[len(merge)-1].args['CARG']}_q", 
                            label=merge[len(merge)-2].label, 
                            args={'ARG0': merge[len(merge)-2].args['ARG0'], 
                                'RSTR': merge[len(merge)-2].args['RSTR'],
                                'BODY': merge[len(merge)-2].args['BODY']},
                            surface=pattern)
                mrs_lf.predications.insert(loc, ep)
                mrs_lf = SemanticParser._remove_eps(mrs_lf, merge)
            elif re.match('not \w*', pattern):
                print(pattern)

        # Add arguments for preddications to match syntax 
        for p in mrs_lf.predications:
            if p.is_quantifier():
                p.predicate +=  "<SPACE>" +p.iv +"."
            elif 'neg' == p.predicate:
                continue
            else:
                arg_str = "("+",".join([ v for v in p.args.values() if not('e' in v or 'i' in v)])+")"
                arg_str = arg_str.replace('u','x')  
                p.predicate += arg_str

        with open(self.tmp_path, 'w+') as f:
            f.write(mrsprolog.encode(mrs_lf))

        parses = subprocess.run(['java', '-jar', self.utool_path, 'solve', '-I', 'mrs-prolog', '-O', 'term-prolog', self.tmp_path ], capture_output=True).stdout.decode('UTF-8')
        os.remove(self.tmp_path)
        
        # cleaning
        parses = parses.splitlines()
        parses = [p[:len(p)-1] if p[len(p)-1] == ',' else p for p in parses ]
        
        pattern = "\S*".join([p.predicate for p in mrs_lf.predications if p.is_quantifier() or p.predicate == 'neg'])
        parse = [p for p in parses if re.search(pattern, p) is not None][0].replace("<SPACE>", " ")
        for in_pred,out_pred in self.transforms:
            parse = re.sub(in_pred,out_pred, parse)

        return parse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLU pipeline')
    parser.add_argument('--grm', default='./external/erg-2018-x86-64-0.9.31.dat', help='Path to grammar')
    parser.add_argument('--ace', default='./external/ace', help='Path to ACE')
    parser.add_argument('--utool', default='./external/Utool-3.1.1.jar', help='Path to Utool jar file')
    parser.add_argument('--ref_exp', default='a circle not behind the square.', help='NL string to parse')

    args = parser.parse_args()

    nlu = SemanticParser(args.grm, args.ace)

    lf = nlu(args.ref_exp)

    print(lf)
    