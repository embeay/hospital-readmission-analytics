"""Logistic regression v2: matched cohort/features/splits to XGBoost v2; nominal admin codes."""
import json,pathlib,numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from sklearn.calibration import calibration_curve
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'logistic_v2_results';OUT.mkdir(exist_ok=True)
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
pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=50))]),cat)],sparse_threshold=.3)
model=Pipeline([('pre',pre),('logit',LogisticRegression(C=1,max_iter=1200,solver='lbfgs'))])
model.fit(X.iloc[tr],y.iloc[tr]);summary={'description':'Logistic v2: nominal administrative identifiers; matched cohort and split to XGBoost v2. Numeric predictors standardized; L2 regularization. No reference category omitted: dummy coefficients are not standalone category-vs-reference odds ratios.','n_total':len(df),'events_total':int(y.sum()),'numeric_features':num,'categorical_features':cat,'encoded_feature_count':len(model.named_steps['pre'].get_feature_names_out())}
for label,idx in [('train',tr),('validation',va),('test',te)]:
 p=model.predict_proba(X.iloc[idx])[:,1];truth=y.iloc[idx].to_numpy();order=np.argsort(-p)
 s={'n':len(idx),'patients':int(groups.iloc[idx].nunique()),'events':int(truth.sum()),'event_rate':float(truth.mean()),'roc_auc':float(roc_auc_score(truth,p)),'pr_auc':float(average_precision_score(truth,p)),'brier':float(brier_score_loss(truth,p)),'log_loss':float(log_loss(truth,p))}
 for frac in [.1,.2]:
  n=int(np.ceil(len(idx)*frac));sel=truth[order[:n]];k=f'top_{int(frac*100)}pct';s[k+'_precision']=float(sel.mean());s[k+'_recall']=float(sel.sum()/truth.sum());s[k+'_events']=int(sel.sum())
 summary[label]=s
 if label=='test':
  pd.DataFrame({'encounter_id':df.iloc[idx].encounter_id.to_numpy(),'actual':truth,'predicted':p}).to_csv(OUT/'test_predictions.csv',index=False)
  obs,pred=calibration_curve(truth,p,n_bins=10,strategy='quantile');pd.DataFrame({'mean_predicted_probability':pred,'observed_event_rate':obs}).to_csv(OUT/'test_calibration.csv',index=False)
terms=model.named_steps['pre'].get_feature_names_out();coef=model.named_steps['logit'].coef_[0]
pd.DataFrame({'encoded_feature':terms,'coefficient':coef,'exp_coefficient':np.exp(np.clip(coef,-30,30))}).sort_values('coefficient',ascending=False).to_csv(OUT/'coefficients.csv',index=False)
summary['converged']=bool(model.named_steps['logit'].n_iter_[0]<1200)
(OUT/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:summary[k] for k in ['encoded_feature_count','converged','validation','test']},indent=2))
