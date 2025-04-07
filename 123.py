import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
data = pd.read_excel("C:/Users/YangD/Desktop/fleet/all_departure.xlsx")
data = data[["Flight_Number", 'Destination_Airport', "Departure_delay", "Delay_Carrier", "Delay_Weather",
             "Delay_National_Aviation_System", "Delay_Security", "Tail_Number"]]
data.dropna(inplace=True)
data["Departure_delay"] = (data["Departure_delay"] > 15) * 1
le = LabelEncoder()
data['Tail_Number'] = le.fit_transform(data['Tail_Number'])
scaler = StandardScaler()
data['Tail_Number'] = scaler.fit_transform(data['Tail_Number'].values.reshape(-1, 1))
cols = ["Flight_Number", 'Destination_Airport']
for item in cols:
    data[item] = data[item].astype('category').cat.codes + 1
x_train, x_temp, y_train, y_temp = train_test_split(data.drop(['Departure_delay'], axis=1),
                                                    data['Departure_delay'], random_state=42, train_size=0.30)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, random_state=42, test_size=0.5)
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(x_test, label=y_test)
params = {
    'max_depth': 10,
    'eta': 0.05310192652727056,
    'gamma': 0.1445222060182382,
    'subsample': 0.7152942553229076
}
model_xgb = xgb.train(params, d_train, num_boost_round=100)
y_pred_test = model_xgb.predict(d_test)
y_pred_train = model_xgb.predict(d_train)
Score_train = roc_auc_score(y_train, (y_pred_train > 0.5).astype(int))
Score_test = roc_auc_score(y_test, (y_pred_test > 0.5).astype(int))
# 计算评估指标
y_pred_test_binary = (y_pred_test > 0.5).astype(int)
y_pred_train_binary = (y_pred_train > 0.5).astype(int)  # 新增训练集二值化

# 计算各项指标
Score_train = roc_auc_score(y_train, y_pred_train_binary)  # 保持原计算方式
Score_test = roc_auc_score(y_test, y_pred_test_binary)

precision_train = precision_score(y_train, y_pred_train_binary)
recall_train = recall_score(y_train, y_pred_train_binary)
f1_train = f1_score(y_train, y_pred_train_binary)

precision_test = precision_score(y_test, y_pred_test_binary)
recall_test = recall_score(y_test, y_pred_test_binary)
f1_test = f1_score(y_test, y_pred_test_binary)

# 打印评估结果
print(f"training_set_AUC Score: {Score_train:.4f}")
print(f"training_set_Precision: {precision_train:.4f}")
print(f"training_set_Recall:    {recall_train:.4f}")
print(f"training_set_F1 Score:  {f1_train:.4f}")
print(f"testing_set_AUC Score: {Score_test:.4f}")
print(f"testing_set_Precision: {precision_test:.4f}")
print(f"testing_set_Recall:    {recall_test:.4f}")
print(f"testing_set_F1 Score:  {f1_test:.4f}")
feature_importance = model_xgb.get_score(importance_type='weight')
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
importances_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.savefig('Importance Score.jpg')
plt.show()
y_pred_test_binary = (y_pred_test > 0.5).astype(int)
conf_mat = confusion_matrix(y_test, y_pred_test_binary)
conf_mat_df = pd.DataFrame(conf_mat, index=['Actual Negative', 'Actual Positive'],
                          columns=['Predicted Negative', 'Predicted Positive'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_df, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('ConfusionMatrix.jpg')
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Curve.jpg')
plt.show()

