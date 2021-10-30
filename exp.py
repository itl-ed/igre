from typing import Dict, List
from argparse import ArgumentParser
import random
from datetime import datetime
from collections import defaultdict, Counter
from functools import reduce
from igre.logic.model import DomainModel, Denotation, Entities

import torch
from torch import  Tensor
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import wandb

from igre.utils import ShapeWorldDataset, create_model_init
from igre.grounder import Grounder
from igre.reasoner import Reasoner, HeadOnlyReasoner, ExistentialReasoner
from igre.lang import RefExpParser
from igre.logic import LogicRefExpParser


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
            idxs = (values_pred >= threshold).nonzero()
            for idx in idxs:
                extension[grounder.symbols[idx[1]]].add(denotation)
        return extension

    prop_extension = discretize(prop_features, prop_grounder, prop_threshold)
    rels_extension = discretize(rels_features, rels_grounder, rels_threshold)

    extension = defaultdict(set, {**prop_extension, **rels_extension})

    return DomainModel(entities, extension)

def metrics(model_true: DomainModel,
            model_pred: DomainModel,
            model_init: DomainModel) -> None:

    """predicted model evaluation"""
    print("========================================")
    print("Evaluation")
    print(f"Ground-truth domain model: {model_true}")
    print(f"Predicted domain model: {model_pred}")
    print("========================================")
    print("Extrinsic evaluation")
    atoms_init = sum([list(x) for x in model_init.invert().values()],[])

    atoms_true = model_true.invert()
    atoms_true = [[x for x in xs if x not in atoms_init] for xs in atoms_true.values()]

    atoms_pred = model_pred.invert()
    atoms_pred = [list(atoms_pred[denotation]) for denotation, atoms in model_true.invert().items()]
    atoms_pred = [[x for x in xs if x not in atoms_init] for xs in atoms_pred]
    mlb = MultiLabelBinarizer().fit(atoms_pred+atoms_true)
    atoms_true = mlb.transform(atoms_true)
    atoms_pred = mlb.transform(atoms_pred)
    summ = classification_report(atoms_true, atoms_pred, zero_division=True)
    print(summ)

    # print(f"Intrinsic evaluation")
    # refexps = [parser(refexp) for refexp in refexps]
    # den_true = [[model_true.denotations(refexp.snt, refexp.var) for refexp in refexps]]
    # den_pred = [[model_pred.denotations(refexp.snt, refexp.var) for refexp in refexps]]
    # mlb = MultiLabelBinarizer().fit(den_pred + den_true)
    # print(den_true, den_pred)
    # den_true = mlb.transform(den_true)
    # den_pred = mlb.transform(den_pred)
    # summ = classification_report(den_true, den_pred, zero_division=True)
    # print(summ)    


def evaluation(config, dataloader, prop_grounder, rels_grounder):

    for prop_features,rels_features,refexps, _ , model_true in dataloader:

        atoms_true = model_true.invert()
        
        for denotation, feature in prop_features.items():
            prop_pred = prop_grounder.predict(feature)

        model_init = create_model_init(model_true, args.known)
        model_pred = prediction(model_true.entities,
                            prop_features,
                            prop_grounder,
                            config.prop_threshold,
                            rels_features,
                            rels_grounder,
                            config.rels_threshold)

        if args.test_evaluation:
            metrics(model_true, model_pred, model_init)


def main(config):

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    language = RefExpParser(config.grm_path,config.ace_path,config.utool_path)
    logic = LogicRefExpParser()
    parser = lambda x: logic(language(x))

    train = ShapeWorldDataset(data_path = config.train_path,
                            feature_extractor = config.feature_extractor,
                            img_size = config.img_size,
                            num_worlds = config.train_num_worlds,
                            num_refexp = config.train_num_ref_exp,
                            num_corr = config.train_num_corrections,
                            shuffle = config.train_shuffle)

    test = ShapeWorldDataset(data_path = config.test_path,
                            feature_extractor = config.feature_extractor,
                            img_size = config.img_size,
                            num_worlds = config.test_num_worlds,
                            num_refexp = config.test_num_ref_exp,
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

    for prop_features, rels_features, refexps, corrs, model_true in train:

        # re-index entitities in the domain model
        total_obs, index = model_true.reindex(total_obs)
        # property key update
        prop_features = {index[k]:v for k, v in prop_features.items()}
        prop_grounder.add_keys(prop_features)
        # relationship key update
        rels_features = {index[k]:v for k, v in rels_features.items()}
        rels_grounder.add_keys(rels_features)

        # create reasoner for a new situation
        model_init = create_model_init(model_true, config.known)

        if config.use_headonly_reasoner:
            reasoner = HeadOnlyReasoner(model = model_init, addmc_path = config.addmc_path)
        elif config.use_existential_reasoner:
            reasoner = ExistentialReasoner(model = model_init, addmc_path = config.addmc_path)
        else:
            reasoner = Reasoner(model = model_init, addmc_path = config.addmc_path)

        for refexp in refexps:

            train_iter += 1
            wandb.log({"referential expression":refexp})
            refexp = parser(refexp)
            
            prop_grounder.add_symbols([s for s,a in refexp.symbols if a == 1])
            rels_grounder.add_symbols([s for s,a in refexp.symbols if a == 2])

            values = reasoner.estimate()

            prop_values = {k:v for k,v in values.items() if k.arity == 1 and k.name in prop_grounder.symbols}
            rels_values  = {k:v for k,v in values.items() if k.arity == 2 and k.name in rels_grounder.symbols}

            prop_grounder.update_values(prop_values)
            rels_grounder.update_values(rels_values)

            model_pred = prediction(model_true.entities,
                                prop_features,
                                prop_grounder,
                                config.prop_threshold,
                                rels_features,
                                rels_grounder,
                                config.rels_threshold)
            
            if config.train_evaluation:
                metrics(model_true, model_pred, model_init)
            evaluation(config, test, prop_grounder, rels_grounder)

            reasoner.add_refexp(refexp, model_true)
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

        for corr,referant in corrs:

            train_iter += 1
            # get logical form
            corr  = parser(corr)
            snt = corr.to_snt([referant])
            reasoner.add_sentence(snt)
            # batch update
            if train_iter != 1 and train_iter % args.batch_freq == 0:
                prop_grounder.batch_learning_mode(epochs = config.batch_epochs,
                                        batch_size = config.batch_size,
                                        shuffle = config.batch_shuffle,
                                        report_freq = config.batch_report_freq)

                rels_grounder.batch_learning_mode(epochs = config.batch_epochs,
                                        batch_size = config.batch_size,
                                        shuffle = config.batch_shuffle,
                                        report_freq = config.batch_report_freq)

            evaluation(config, test, prop_grounder, rels_grounder)

if __name__ == '__main__':

    parser = ArgumentParser("IGRE experiments")
    parser.add_argument("--log", type=str, help="logging location")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    # model 
    parser.add_argument("--feature-extractor", default="densenet161", help="feature extractor to use for images")
    parser.add_argument("--img-size", default=64, type=int, help="size of the image")
    parser.add_argument("--addmc-path", default="./external/addmc", help="path to ADDMC weigthed model counter")
    parser.add_argument("--known", nargs='*',default=[], help='list of known symbols')
    parser.add_argument("--use-headonly-reasoner", default=False, action="store_true", help="if true, only use head for estimation")
    parser.add_argument("--use-existential-reasoner", default=False, action="store_true", help="if true use existential reasoner")
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
    parser.add_argument("--train-num-corrections", default=5, type=int, help="number of corrections to use")
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
    wandb.init(project='igre', entity='rimvydasrub', name=str(datetime.now()))
    config = wandb.config
    config.update(args)

    main(config)