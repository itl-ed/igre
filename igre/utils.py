import os
from typing import Dict, List, Tuple
import json, random, csv
from collections import defaultdict

from PIL import Image
import torch
from torchvision import transforms, models

from igre.logic.syntax import LogicRefExpParser, RefExp
from igre.lang import RefExpParser
from igre.logic import DomainModel


class ShapeWorldDataset:
    """Shapeworld based visual reference resolution Dataset"""

    def __init__(self,
                data_path:str,
                feature_extractor:str,
                img_size:int,
                num_worlds:int,
                num_refexp:int,
                grm_path: str,
                ace_path: str,
                utool_path: str,
                shuffle: bool = True,):
        
        self.data_path = data_path
        self.extractor = getattr(models, feature_extractor)(pretrained=True)
        self.extractor.eval()
        self.img_size = img_size
        self.num_worlds = num_worlds
        self.num_refexp = num_refexp
        self.grm_path = grm_path
        self.ace_path = ace_path
        self.utool_path = utool_path
        self.idx2sample = list(range(self.num_worlds))
        if shuffle:
            random.shuffle(self.idx2sample)

        self.preprocess = transforms.Compose([
                            transforms.Resize(self.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                [0.229, 0.224, 0.225])])

        self.language = RefExpParser(self.grm_path,self.ace_path,self.utool_path)
        self.logic = LogicRefExpParser()
        # self.parser = lambda x: logic(language(x))


    def __get_position(self,
                    model_path:str) -> Dict[int,Tuple]:
        """
        Get position point coordinates for each entitiy in the visual scene:
        :param model_path: path to model file
        
        :return dict of entity_id: point tuple"""

        with open(model_path) as file:
            entities = json.load(file)['entities']

        return {e['id']: (self.img_size*e['bounding_box']['topleft']['x'],
                        self.img_size*e['bounding_box']['topleft']['y'],
                        self.img_size*e['bounding_box']['bottomright']['x'],
                        self.img_size*e['bounding_box']['bottomright']['y'],
                        self.img_size*e['center']['x'],
                        self.img_size*e['center']['y'])
                        for e in entities}

    def __get_refexps(self,refexp_path: str) -> List[RefExp]:
        """Get num_refexp from the refexp file"""
        with open(refexp_path, 'r') as file:
            refexps = [l.strip() for l in file.readlines()]
        refexps = random.sample(refexps, k=self.num_refexp)

        return [self.logic(self.language(refexp)) for refexp in refexps]

    def __get_features(self, world_path:str, positions:Dict) -> Dict:
        """Extract first and second order features using bboxes"""

        img = Image.open(world_path)

        prop = {}
        ## first-order features by bounding boxes
        for entity, position in positions.items():
            
            crop = position[:4]

            pos = torch.Tensor(position).reshape((1,6))

            # [1, input_size] <- CNN([1, 3, img_size, img_size])
            features = self.preprocess(img.crop(crop)).unsqueeze(0)
            with torch.no_grad():
                features = self.extractor(features)
            prop[entity] = torch.hstack((features, pos)) 

        rel = {}
        ## second-order 
        for entity1, feat1 in prop.items():
            for entity2, feat2 in prop.items():
                rel[(entity1,entity2)] = torch.hstack((feat1, feat2))

        return prop, rel 

    @staticmethod
    def __get_model(model_path: str, threshold: float = 0.1) -> DomainModel:
        """make DomainModel from JSON file
        :param model_path: path to JSON 
        :param threshold: threshold for determining spatial relationship
        
        :return domain model"""
        with open(model_path, 'r') as file:
            model = json.load(file)
        
        domain = set(e['id'] for e in model['entities'])
        valuation = defaultdict(set)
        for e in model['entities']:
            valuation[e['color']['name']].add(e['id'])
            valuation[e['shape']['name']].add(e['id'])
            valuation['object'].add(e['id'])
        
        for u in model['entities']:
            for v in model['entities']:

                e = (u['id'],v['id'])
                dx = u['center']['x'] - v['center']['x'] 
                dy = u['center']['y'] - v['center']['y']
                if dx > threshold:
                    valuation['right'].add(e)
                if dx < -threshold:
                    valuation['left'].add(e)
                if dy < -threshold:
                    valuation['above'].add(e)
                if dy > threshold:
                    valuation['below'].add(e)
        
        return DomainModel(domain, valuation)

    def __len__(self):
        """len is number of situations to consider."""
        return self.num_worlds

    # def __get_corrections(self, corr_path: str) -> List[Tuple[str,int]]:

    #     if not os.path.isfile(corr_path):
    #         return None
    #     with open(corr_path, 'r') as file:
    #         reader = csv.reader(file, delimiter=',',)
    #         #remove heaher
    #         next(reader)
    #         corrs = [(row[0].strip(), int(row[1]))  for row in reader]

    #         return random.sample(corrs, k=self.num_corr)



    def __getitem__(self, idx:int) -> Tuple[Dict,List,Dict,DomainModel]:
        """Single item consists of :
        - first and second order features of enitities in the scene
        - referencial expressions applied
        - ground truth domain model
        """
        idx = self.idx2sample[idx]

        world_path = f'{self.data_path}/world-{idx:02d}/world-{idx:02d}.bmp'
        model_path = f'{self.data_path}/world-{idx:02d}/model-{idx:02d}.json'
        refexp_path = f'{self.data_path}/world-{idx:02d}/refexp-{idx:02d}.txt'
        # corr_path = f'{self.data_path}/world-{idx:02d}/corrections-{idx:02d}.csv'

        positions = self.__get_position(model_path)
        prop, rel = self.__get_features(world_path, positions)
        refexps = self.__get_refexps(refexp_path)
        # corrections = self.__get_corrections(corr_path)
        model = ShapeWorldDataset.__get_model(model_path)

        return prop, rel, refexps, model



def create_model_init(model_true: DomainModel, known: List[str]) -> DomainModel:
    ext = {k:v for k,v in model_true.extension.items() if k in known}
    ext = defaultdict(set,ext)
    return DomainModel(model_true.entities, ext)


# def evaluate_model(model_true: DomainModel,
#                 model_pred: DomainModel,
#                 model_init: DomainModel,
#                 flag: str,
#                 log: str) -> None:
#     """Evaluate domain model prediction
#     :param gold_model: ground-truth model
#     :param pred_model: predicted domain model
#     :param init_model: already known atoms
#     :param flag: annotation flag
#     :param log: logging directory"""

#     # evaluate properties:
#     def symbols(model: DomainModel) -> defaultdict:
#         symbols = defaultdict(set)
#         for symbol, denotations in model_true.extension.items():
#             if symbol in model_init.extension.keys():
#                 continue
#             for denotation in denotations:
#                 symbols[denotation].add(symbol)
#         return symbols

    
#     symbols_pred  = symbols(model_pred)
#     symbols_true  = symbols(model_true)
    
#     mlb = MultiLabelBinarizer(sparse_output=True).fit({**symbols_pred, **symbols_true})

#     symbol_pred_mlb = mlb.transform(symbols_pred)
#     symbol_true_mlb = mlb.transform(symbols_true)

#     clas

    # symbol_pred = defaultdict(set)
    # for symbol, denotations in model_pred.extension.items():
    #     if symbol in model_init.extension.keys():
    #         continue
    #     for denotation in denotations:
    #         symbol_pred[denotation].add(symbol)



    # props, rels = [], []
    # for key, values in true_model.extension.items():

    #     if key not in init_model.extension.keys():
    #         if all(isinstance(value, int) for value in values):
    #             props.append(key)
    #         elif  all(isinstance(value, tuple) for value in values):
    #             rels.append(key)
    
    # ## evaluate props
    
    # gold = {u:[] for u in true_model.entities}
    # pred = {u:[] for u in true_model.entities}

    # for u in true_model.entities:

    #     for prop in props:
            
    #         if prop in true_model.extension.keys() and u in true_model.extension[prop]:
    #             gold[u].append(prop)

    #         if prop in pred_model.extension.keys() and u in pred_model.extension[prop]:
    #             pred[u].append(prop)

    # gold = list(gold.values())
    # pred = list(pred.values())

    # mlb = MultiLabelBinarizer().fit([props])
    # gold = mlb.transform(gold)
    # pred = mlb.transform(pred)
    # cms = multilabel_confusion_matrix(gold,pred)

    # for idx, cm in enumerate(cms):
    #     plt.clf()
    #     plt.figure(figsize=(10,10))
    #     pred_labels = [f"PRED: ¬{props[idx]}", f"PRED: {props[idx]}"]
    #     true_labels = [f"TRUE: ¬{props[idx]}", f"TRUE: {props[idx]}"]
    #     df_cm = pd.DataFrame(cm,  index=true_labels , columns=pred_labels)
    #     sns.set(font_scale=1.4) # for label size
    #     sns.heatmap(df_cm, 
    #                 annot=True,
    #                 annot_kws={"size": 10}, # font size
    #                 cbar=False) 
    #     plt.savefig(f'{log}/{flag}-{props[idx]}.png')
    #     plt.close()


    # # evaluate rels
    # gold = {u:[] for u in product(true_model.entities, repeat=2)}
    # pred = {u:[] for u in product(true_model.entities, repeat=2)}

    # for u in product(true_model.entities, repeat=2):

    #     for rel in rels:
            
    #         if rel in true_model.extension.keys() and u in true_model.extension[rel]:
    #             gold[u].append(rel)

    #         if rel in pred_model.extension.keys() and u in pred_model.extension[rel]:
    #             pred[u].append(rel)

    # gold = list(gold.values())
    # pred = list(pred.values())

    # mlb = MultiLabelBinarizer().fit([rels])
    # gold = mlb.transform(gold)
    # pred = mlb.transform(pred)
    # cms = multilabel_confusion_matrix(gold,pred)

    # for idx, cm in enumerate(cms):
    #     plt.clf()
    #     plt.figure(figsize=(10,10))
    #     pred_labels = [f"PRED: ¬{rels[idx]}", f"PRED: {rels[idx]}"]
    #     true_labels = [f"TRUE: ¬{rels[idx]}", f"TRUE: {rels[idx]}"]
    #     df_cm = pd.DataFrame(cm,  index=true_labels , columns=pred_labels)
    #     sns.set(font_scale=1.4) # for label size
    #     sns.heatmap(df_cm, 
    #                 annot=True,
    #                 annot_kws={"size": 10}, # font size
    #                 cbar=False) 
    #     plt.savefig(f'{log}/{flag}-{rels[idx]}.png')
    #     plt.close()