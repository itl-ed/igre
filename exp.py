from typing import Dict, List, Tuple
from argparse import ArgumentParser
import random
from datetime import datetime
from collections import defaultdict
from itertools import product
from igre.logic.model import DomainModel, Denotation, Entities

import torch
from torch import  Tensor

import wandb
from igre.logic.syntax import AtomicSentence, NegatedSentence, OrSentence, RefExp

from igre.utils import ShapeWorldDataset
from igre.grounder import Grounder
from igre.reasoner import Reasoner
from igre.logic import LogicRefExpParser
from igre.lang import RefExpParser





def prediction(entities: Entities,
        prop_features:  Dict[Denotation,Tensor],
        prop_grounder: Grounder,
        prop_threshold: float,
        rels_features: Dict[Denotation,Tensor],
        rels_grounder: Grounder,
        rels_threshold: float,
        ) -> DomainModel:
    """Domain model prediction using grounders"""

    def discretize(features, grounder, threshold):
        extension = defaultdict(set)
        for denotation, feature in features.items():
            values_pred = grounder.predict(feature)
            idxs = torch.nonzero(values_pred >= threshold)
            for idx in idxs:
                extension[grounder.symbols[idx[1]]].add(denotation)
        return extension

    prop_extension = discretize(prop_features, prop_grounder, prop_threshold)
    rels_extension = discretize(rels_features, rels_grounder, rels_threshold)

    extension = defaultdict(set, {**prop_extension, **rels_extension})

    return DomainModel(entities, extension)

def metrics(model_true: DomainModel,
            model_pred: DomainModel,
            refexps: List[RefExp]) -> Dict[str, float]:
    """evaluation metrics for 
    precision, recall, f1 for model and reference prediction"""
    C1 = ['red', 'blue', 'magenta', 'yellow', 'green', 'cyan']
    S1 = ['square', 'circle', 'triangle', 'pentagon', 'cross', 'ellipse', 'semicircle']
    R2 = ['left', 'right', 'above', 'below']

    def prf(pred: set, real: set) -> Tuple[float, float, float]:

        tp = pred.intersection(real)
        p = len(tp)/len(pred) if len(pred) > 0 else 0
        r = len(tp)/len(real) if len(real) > 0 else 0
        f = 2*p*r/(p+r) if p+r > 0 else 0

        return p, r, f

    def save_avg(elems: List[float]) -> float:

        return sum(elems)/len(elems) if len(elems) > 0 else 0

    def eval_fpr(model_pred, model_true, symbols: List[str]):

        model_ps, model_rs, model_fs = [],[],[]
        for symbol in model_pred.extension.keys():

            if symbol not in symbols:
                continue

            predicted = model_pred.extension[symbol]
            ground = model_true.extension[symbol]

            p,r, f = prf(predicted, ground)
            model_ps.append(p)
            model_rs.append(r)
            model_fs.append(f)

        return model_ps, model_rs, model_fs

    full_ps, full_rs, full_fs = eval_fpr(model_pred, model_true, C1+S1+R2)
    c1_ps, c1_rs, c1_fs = eval_fpr(model_pred, model_true, C1)
    s1_ps, s1_rs, s1_fs = eval_fpr(model_pred, model_true, S1)
    r2_ps, r2_rs, r2_fs = eval_fpr(model_pred, model_true, R2)

    ref_ps, ref_rs, ref_fs = [],[],[]

    for refexp in refexps:

        predicted = model_true.denotations(refexp.snt, refexp.var)
        ground = model_pred.denotations(refexp.snt, refexp.var)
        p,r, f = prf(predicted, ground)
        ref_ps.append(p)
        ref_rs.append(r)
        ref_fs.append(f)

    return {f"ful_p": save_avg(full_ps), 
            f"ful_r": save_avg(full_rs),
            f"ful_f": save_avg(full_fs),
            f"c1_p": save_avg(c1_ps), 
            f"c1_r": save_avg(c1_rs),
            f"c1_f": save_avg(c1_fs),
            f"s1_p": save_avg(s1_ps), 
            f"s1_r": save_avg(s1_rs),
            f"s1_f": save_avg(s1_fs),
            f"r2_p": save_avg(r2_ps), 
            f"r2_r": save_avg(r2_rs),
            f"r2_f": save_avg(r2_fs),
            f"ref_p": save_avg(ref_ps), 
            f"ref_r": save_avg(ref_rs),
            f"ref_f": save_avg(ref_fs),
            }


def evaluation(config: wandb.Config,
            dset: ShapeWorldDataset,
            prop_grounder: Grounder,
            rels_grounder: Grounder,) -> None:

    results = {f"ful_p": 0, 
            f"ful_r": 0,
            f"ful_f": 0,
            f"c1_p": 0, 
            f"c1_r": 0,
            f"c1_f": 0,
            f"s1_p": 0, 
            f"s1_r": 0,
            f"s1_f": 0,
            f"r2_p": 0, 
            f"r2_r": 0,
            f"r2_f": 0,
            f"ref_p": 0, 
            f"ref_r": 0,
            f"ref_f": 0,
            }

    for prop_features,rels_features,refexps, model_true in dset:

        model_pred = prediction(model_true.entities,
                            prop_features,
                            prop_grounder,
                            config.prop_threshold,
                            rels_features,
                            rels_grounder,
                            config.rels_threshold)

        model_results = metrics(model_true, model_pred, refexps)

        for result in results.keys():
            results[result] += model_results[result]

    results = {k:v/len(dset) for k,v in results.items()}

    wandb.log(results)


def main(config):

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    lang = RefExpParser(grm_path=config.grm_path,
                        ace_path=config.ace_path,
                        utool_path=config.utool_path)
    logic = LogicRefExpParser()
    parser = lambda x: logic(lang(x))

    train = ShapeWorldDataset(data_path = config.train_path,
                            feature_extractor = config.feature_extractor,
                            img_size = config.img_size,
                            num_worlds = config.train_num_worlds,
                            num_refexp = config.train_num_ref_exp,
                            grm_path=config.grm_path,
                            ace_path=config.ace_path,
                            utool_path=config.utool_path,
                            shuffle = config.train_shuffle)

    test = ShapeWorldDataset(data_path = config.test_path,
                            feature_extractor = config.feature_extractor,
                            img_size = config.img_size,
                            num_worlds = config.test_num_worlds,
                            num_refexp = config.test_num_ref_exp,
                            grm_path=config.grm_path,
                            ace_path=config.ace_path,
                            utool_path=config.utool_path,
                            shuffle = config.test_shuffle)

    prop_grounder = Grounder(input_size = config.prop_input_size,
                            emb_size = config.prop_emb_size,
                            supp_size = config.prop_supp_size,
                            arity = 1,
                            threshold = config.prop_threshold)
    
    rels_grounder = Grounder(input_size = config.rels_input_size,
                        emb_size = config.rels_emb_size,
                        supp_size = config.rels_supp_size,
                        arity = 2,
                        threshold = config.rels_threshold)
    
    train_iter, total_obs = 0, 0

    for idx, (prop_features, rels_features, refexps, model_true) in enumerate(train):

        print(f"domain model #{idx}")

        # re-index entitities in the domain model
        total_obs, index = model_true.reindex(total_obs)
        # property key update
        prop_features = {index[k]:v for k, v in prop_features.items()}
        prop_grounder.add_keys(prop_features)
        # relationship key update
        rels_features = {index[k]:v for k, v in rels_features.items()}
        rels_grounder.add_keys(rels_features)

        # create reasoner for a new situation
        reasoner = Reasoner(addmc_path = config.addmc_path, 
                        model=DomainModel(model_true.entities))

        for refexp in refexps:

            train_iter += 1
            wandb.log({"referential expression": wandb.Html(str(refexp))})
            
            prop_grounder.add_symbols([s for s,a in refexp.symbols if a == 1])
            rels_grounder.add_symbols([s for s,a in refexp.symbols if a == 2])

            values = reasoner.estimate()

            prop_values = {k:v for k,v in values.items() if k.arity == 1 and k.name in prop_grounder.symbols}
            rels_values  = {k:v for k,v in values.items() if k.arity == 2 and k.name in rels_grounder.symbols}

            prop_grounder.update_values(prop_values)
            rels_grounder.update_values(rels_values)
            
            evaluation(config, test, prop_grounder, rels_grounder)
            reasoner.add_full(refexp, model_true)
            ## add axioms for relations
            for rel in rels_grounder.symbols:
                for term in product(model_true.entities, repeat=1):
                    terms = [term[0], term[0]]
                    irr = NegatedSentence(AtomicSentence(rel, terms))
                    reasoner.add_sentence(irr)

                for terms in product(model_true.entities, repeat=2):
                    pos_terms = list(terms)
                    neg_terms = list(reversed(terms))
                    antisym = OrSentence(left=NegatedSentence(AtomicSentence(rel, pos_terms)),
                    right=NegatedSentence(AtomicSentence(rel, neg_terms))) 

                    reasoner.add_sentence(antisym)

            # batch update
            if train_iter != 1 and train_iter % config.batch_freq == 0:

                prop_grounder.batch_learning_mode(epochs = config.batch_epochs,
                                        batch_size = config.batch_size,
                                        shuffle = config.batch_shuffle,
                                        report_freq = config.batch_report_freq)

                rels_grounder.batch_learning_mode(epochs = config.batch_epochs,
                                        batch_size = config.batch_size,
                                        shuffle = config.batch_shuffle,
                                        report_freq = config.batch_report_freq)

if __name__ == '__main__':

    parser = ArgumentParser("IGRE experiments")
    parser.add_argument("--log", type=str, help="logging location")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    # model 
    parser.add_argument("--feature-extractor", default="densenet161", help="feature extractor to use for images")
    parser.add_argument("--img-size", default=64, type=int, help="size of the image")
    parser.add_argument("--addmc-path", default="./external/addmc", help="path to ADDMC weigthed model counter")
    parser.add_argument("--known", nargs='*',default=[], help='list of known symbols')
    ## natural language processing
    parser.add_argument("--grm-path", default="./external/erg-2018-x86-64-0.9.31.dat", help="path to grammar file")
    parser.add_argument("--ace-path", default="./external/ace", help="path to ACE binaries")
    parser.add_argument("--utool-path", default="./external/Utool-3.1.1.jar", help="path to Utool jar file")
    ## properties 
    parser.add_argument("--prop-input-size", default=1006, type=int, help="input feature size for properties ")
    parser.add_argument("--prop-emb-size", default=5, type=int, help='size of the embedding for properties')
    parser.add_argument("--prop-supp-size", default=3, type=int, help="size of the support set when making predictions for properties")
    parser.add_argument("--prop-threshold", default=0.7,type=float, help="threshold when selecting for prediction for properties")
    ## relations
    parser.add_argument("--rels-input-size", default=1006, type=int, help="input feature size for properties ")
    parser.add_argument("--rels-emb-size", default=5, type=int, help='size of the embedding for properties')
    parser.add_argument("--rels-supp-size", default=3, type=int, help="size of the support set when making predictions for properties")
    parser.add_argument("--rels-threshold", default=0.7,type=float, help="threshold when selecting for prediction for properties")
    # train
    parser.add_argument("--train-path", help="path to training data")
    parser.add_argument("--train-num-worlds", default=30,type=int, help="number of worlds to use for training images")
    parser.add_argument("--train-num-ref-exp", default=5,type=int, help="number of referential expressions to use for singe world in training")
    parser.add_argument("--train-shuffle", action="store_true", default=False, help="stuffle the training data")
    parser.add_argument("--train-evaluation", action="store_true", default=False, help="evaluate during training")
    # batch updates
    parser.add_argument("--batch-freq", default=1,type=int, help="number of worlds to observe before the offline training")
    parser.add_argument("--batch-epochs", default=100, type=int, help="number of epochs in offline training")
    parser.add_argument("--batch-size", default=4, type=int, help="size of the training batch offline")
    parser.add_argument("--batch-shuffle", action="store_true", default=False, help="stuffle baches during offline training")
    parser.add_argument("--batch-report-freq", default=10, type=int, help="offline training report frequency")
    # testing
    parser.add_argument("--test-path", help="path to testing data")
    parser.add_argument("--test-num-worlds", default=10,type=int, help="number of worlds to use for testing")
    parser.add_argument("--test-num-ref-exp", default=5,type=int, help="number of referential expressions to use for singe world in testing")
    parser.add_argument("--test-shuffle", action="store_true", default=False, help="stuffle the testing data")
    parser.add_argument("--test-evaluation", action="store_true", default=False, help="evaluate during testing")

    args = parser.parse_args()

    ## initialize experiments
    wandb.login(key="960dbf7b3effe15f5578aa067e9c88d77f076f51")
    wandb.init(project='igre', 
            entity='rimvydasrub',
            group="exp_final",
            dir = "./wandb_tmp/",
            name='full-axioms_'+str(datetime.now()))
    config = wandb.config
    config.update(args)

    main(config)