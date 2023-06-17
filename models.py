import time

from sklearnex import patch_sklearn
patch_sklearn()

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import daal4py as d4p
import dpctl
import pickle



class ModelBase:
    def __init__(self, name):
        self.name = name
    
    def train(self, data_train, target_train, data_test, target_test):
        print(f'[{self.name}]: Training...')
        if isinstance(self.baseModel, d4p._daal4py.gbt_classification_model):
            print('The model is already trained')
            return
        self.baseModel.fit(
            data_train,
            target_train,
            eval_set=[(data_test, target_test)]
        )
        
        if isinstance(self.baseModel, LGBMClassifier):
            self.baseModel = d4p.get_gbt_model_from_lightgbm(self.baseModel.booster_)
        elif isinstance(self.baseModel, XGBClassifier):
            self.baseModel = d4p.get_gbt_model_from_xgboost(self.baseModel.get_booster())
        elif isinstance(self.baseModel, CatBoostClassifier):
            self.baseModel = d4p.get_gbt_model_from_catboost(self.baseModel)
    
    def predict(self, data_test):
        length = len(data_test)
        if length > 1:
            print(f'[{self.name}]: Predict {length} water samples...')
        else:
            print(f'[{self.name}]: Predict 1 water sample...')
        
        
        begin = time.time()
        prediction = (
            d4p.gbt_classification_prediction(nClasses=2)
            .compute(data_test, self.baseModel)
            .prediction
        )
        end = time.time()
        print(f"Inference time (oneAPI): {end - begin:.2f}s") 
        
        return prediction
    
    def save(self, save_path):
        with open(save_path,'wb') as file:
            pickle.dump(self, file)


class LGBM(ModelBase):
    def __init__(self):
        self.name = 'LGBM'
        self.baseModel = LGBMClassifier(
                            objective="binary",
                            n_estimators=30,
                            learning_rate=0.27099464626835873,
                            num_leaves=1300,
                            max_depth=12,
                            min_data_in_leaf=1000,
                            reg_alpha=0.1786310325541849,
                            reg_lambda=1.3163677959254036,
                            min_gain_to_split=5.158405494258322,
                            bagging_fraction=0.8,
                            bagging_freq=1,
                            feature_fraction=0.9,
                            verbosity=-1,
                        )

class XGB(ModelBase):
    def __init__(self):
        self.name = 'XGB'
        self.baseModel = XGBClassifier(
                            max_depth=9,
                            objective="binary:logistic",
                            use_label_encoder=False,
                            eval_metric="logloss",
                            early_stopping_rounds=30,
                            min_child_weight=6,
                            gamma=3,
                            learning_rate=1.0,
                            subsample=0.9955784740143707,
                            colsample_bytree=0.8233654438141474,
                            reg_alpha=9.85268289322318,
                            reg_lambda=57.976004599564774,
                            n_estimators=11,
                        )

class CatBoost(ModelBase):
    def __init__(self):
        self.name = 'CatBoost'
        self.baseModel = CatBoostClassifier(
                            depth=11,
                            objective="Logloss",
                            learning_rate=1.0,
                            eval_metric="Logloss",
                            early_stopping_rounds=30,
                            n_estimators=1800,
                            max_bin=322,
                            min_data_in_leaf=1,
                            l2_leaf_reg=0.4421314590378351,
                            subsample=0.8680019204265281,
                        )