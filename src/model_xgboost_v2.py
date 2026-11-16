"""XGBoost v2: nominal administrative IDs, unchanged patient-group split.
Model selection uses validation; held-out test is evaluated for comparison only.
"""
import json,pathlib,numpy as np,pandas as pd
from scipy.special import expit
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from xgboost import XGBClassifier
ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'xgboost_v2_results';OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT.parent/'diabetic_data.csv').replace('?',np.nan)
df=df[~df.discharge_disposition_id.isin([11,13,14,19,20,21]) & (df.gender!='Unknown/Invalid')].copy()
df['target']=(df.readmitted=='<30').astype(int)
for c in ['diag_1','diag_2','diag_3']:df[c]=df[c].fillna('Missing').astype(str).str[:3]
admin=['discharge_disposition_id','admission_type_id','admission_source_id']
for c in admin:df[c]=df[c].astype('string').fillna('Missing').astype(str)
remove=['encounter_id','patient_nbr','readmitted','target','weight','payer_code','medical_specialty']
features=[c for c in df if c not in remove];X=df[features].copy();y=df.target;groups=df.patient_nbr
num=X.select_dtypes(include='number').columns.tolist();cat=[c for c in features if c not in num]
outer=GroupShuffleSplit(n_splits=1,test_size=.30,random_state=2026)
tr,tmp=next(outer.split(X,y,groups));inner=GroupShuffleSplit(n_splits=1,test_size=.5,random_state=2027)
vr,ter=next(inner.split(X.iloc[tmp],y.iloc[tmp],groups.iloc[tmp]));va=tmp[vr];te=tmp[ter]
pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True),num),('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=50))]),cat)],sparse_threshold=.3)
Xt=pre.fit_transform(X.iloc[tr]);Xv=pre.transform(X.iloc[va]);Xe=pre.transform(X.iloc[te])
model=XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.045,subsample=.85,colsample_bytree=.85,min_child_weight=12,reg_lambda=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=20,tree_method='hist',random_state=2026,n_jobs=2)
model.fit(Xt,y.iloc[tr],eval_set=[(Xv,y.iloc[va])],verbose=False)
def ev(idx,p):
 truth=y.iloc[idx].to_numpy();order=np.argsort(-p);r={'n':len(idx),'events':int(truth.sum()),'roc_auc':float(roc_auc_score(truth,p)),'pr_auc':float(average_precision_score(truth,p)),'brier':float(brier_score_loss(truth,p)),'log_loss':float(log_loss(truth,p))}
 for frac in [.1,.2]:
  n=int(np.ceil(len(idx)*frac));sel=truth[order[:n]];r[f'top_{int(frac*100)}pct_precision']=float(sel.mean());r[f'top_{int(frac*100)}pct_recall']=float(sel.sum()/truth.sum());r[f'top_{int(frac*100)}pct_events']=int(sel.sum())
 return r
pv=model.predict_proba(Xv)[:,1];pe=model.predict_proba(Xe)[:,1]
res={'description':'XGBoost v2: three administrative ID fields nominal categorical, diagnoses categorical; all other features, split, hyperparameters unchanged','numeric_features':num,'categorical_features':cat,'encoded_feature_count':len(pre.get_feature_names_out()),'best_iteration':int(model.best_iteration),'validation':ev(va,pv),'test':ev(te,pe)}
# Historical v1 comparison omitted from standalone frozen v2 reproduction.
# SHAP: use sparse matrix, same representation as scoring; native XGBoost contributions sum to margin.
sv=Xv[:min(1500,Xv.shape[0])]; booster=model.get_booster()
from xgboost import DMatrix
contrib=booster.predict(DMatrix(sv),pred_contribs=True,iteration_range=(0,model.best_iteration+1))
margin=booster.predict(DMatrix(sv),output_margin=True,iteration_range=(0,model.best_iteration+1))
scored=model.predict_proba(sv)[:,1]
err_margin=float(np.max(np.abs(contrib.sum(axis=1)-margin)));err_prob=float(np.max(np.abs(expit(contrib.sum(axis=1))-scored)))
res['shap_checks']={'matrix_sparse':bool(sparse.issparse(sv)),'max_abs_margin_reconstruction_error':err_margin,'max_abs_probability_reconstruction_error':err_prob,'n_checked':sv.shape[0]}
assert err_margin<1e-4 and err_prob<1e-5
names=pre.get_feature_names_out(); orig=[]
for name in names:
 if name.startswith('numeric__'):orig.append(name.split('numeric__',1)[1])
 else:
  tail=name.split('categorical__',1)[1]
  matches=[c for c in cat if tail==c or tail.startswith(c+'_')]
  orig.append(max(matches,key=len) if matches else 'UNMAPPED_'+tail)
assert not any(z.startswith('UNMAPPED') for z in orig)
importance=[]
for f in features:
 inds=[i for i,z in enumerate(orig) if z==f]
 if inds:
  grouped=contrib[:,inds].sum(axis=1)
  importance.append({'feature':f,'encoded_columns':len(inds),'mean_abs_shap_log_odds':float(np.mean(np.abs(grouped))),'mean_signed_shap_log_odds':float(np.mean(grouped)),'share_positive':float(np.mean(grouped>0))})
pd.DataFrame(importance).sort_values('mean_abs_shap_log_odds',ascending=False).to_csv(OUT/'grouped_shap_validation.csv',index=False)
pd.DataFrame({'encounter_id':df.iloc[te].encounter_id.to_numpy(),'actual':y.iloc[te].to_numpy(),'predicted':pe}).to_csv(OUT/'test_predictions.csv',index=False)
pd.DataFrame({'encoded_feature':names,'gain_importance':model.feature_importances_,'original_feature':orig}).sort_values('gain_importance',ascending=False).to_csv(OUT/'encoded_feature_importance.csv',index=False)
(OUT/'metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
