import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from hyperopt import hp, STATUS_OK, Trials, fmin, partial, anneal
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from time import time
start_time = time()
data = pd.read_excel("C:/Users/YangD/Desktop/fleet/all_departure.xlsx", )
data = data[["Flight_Number", 'Destination_Airport',"Scheduled_departure_time",
             "Departure_delay", "Delay_Security", 'Tail_Number',
             "Delay_National_Aviation_System"]]
print('Data_set:', data.shape)
data.dropna(inplace=True)
data["Departure_delay"] = (data["Departure_delay"] > 15) * 1
le = LabelEncoder()
data['Tail_Number'] = le.fit_transform(data['Tail_Number'])
scaler = StandardScaler()
data['Tail_Number'] = scaler.fit_transform(data['Tail_Number'].values.reshape(-1, 1))
cols = ["Flight_Number", 'Destination_Airport',"Scheduled_departure_time"]
for item in cols:
    data[item] = data[item]. astype('category').cat.codes + 1
x_train, x_test, y_train, y_test = train_test_split(data.drop(['Departure_delay'], axis=1),
                                                    data['Departure_delay'], random_state=42, train_size=0.60)
cate_features_name = ["Flight_Number", 'Destination_Airport', "Scheduled_departure_time",
                      "Delay_Security", "Tail_Number", "Delay_National_Aviation_System"]
d_train = xgb.DMatrix(x_train, label=y_train, feature_names=cate_features_name)
d_test = xgb.DMatrix(x_test, feature_names=cate_features_name)
# 贝叶斯优化函数
def bayes_fmin(train_x, test_x, train_y, test_y, eval_iters=50):
    def factory(params):
        fit_params = {
            'max_depth': int(params['max_depth']),
            'learning_rate': params['eta'],
            'subsample': params['subsample'],
            'gamma': params['gamma'],
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'early_stopping_rounds': 10,
        }
        model = XGBClassifier(**fit_params)
        model.fit(
            train_x, train_y,
            eval_set=[(test_x, test_y)],
            verbose=False
        )
        y_pred_proba = model.predict_proba(test_x)[:, 1]
        auc_score = roc_auc_score(test_y, y_pred_proba)
        return {'loss': -auc_score, 'status': STATUS_OK}

    space = {
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'eta': hp.loguniform('eta', -5, 0),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'gamma': hp.uniform('gamma', 0, 5)
    }

    best_params = fmin(
        fn=factory,
        space=space,
        algo=partial(anneal.suggest),
        max_evals=eval_iters,
        trials=Trials(),
        return_argmin=True
    )
    best_params['max_depth'] = int(best_params['max_depth'])
    return best_params

# 执行优化
best_params = bayes_fmin(x_train, x_test, y_train, y_test, 100)
print("Best Parameters:", best_params)