import time

from sklearnex import patch_sklearn
patch_sklearn()

from lightgbm import LGBMClassifier
import daal4py as d4p
import dpctl
import pickle



class ModelBase:
    def __init__(self, name):
        self.name = name
        
    def load(self):
        pass
    
    def train(self, train_data):
        pass        
    
    def predict(self, data_test):
        pass
    
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
        
    
    def load(self):
        print(f'[{self.name}]: Load Model...')

    def train(self, data_train, target_train, data_test, target_test):
        print(f'[{self.name}]: Training...')
        if isinstance(self.baseModel, d4p._daal4py.gbt_classification_model):
            print('The model is already trained')
            return
        self.baseModel.fit(
            data_train,
            target_train,
            eval_set=[(data_test, target_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
        )
        self.baseModel = d4p.get_gbt_model_from_lightgbm(self.baseModel.booster_)
    
    def predict(self, data_test):
        length = len(data_test)
        if length > 1:
            print(f'[{self.name}]: Predict {length} water samples...')
        else:
            print(f'[{self.name}]: Predict 1 water sample...')
        
        
        begin = time.time()
        daal_lgbm_model_prediction = (
            d4p.gbt_classification_prediction(nClasses=2)
            .compute(data_test, self.baseModel)
            .prediction
        )
        end = time.time()
        print(f"GPU Inference time (oneAPI): {end - begin:.2f}s") 
        
        return daal_lgbm_model_prediction