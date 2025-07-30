import tushare as ts
import numpy as np
import pandas as pd
import time
import random
import os
import tensorflow as tf
from datetime import datetime,timedelta

seed_value=42
os.environ["PYTHONHASHSEED"]=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
features=[f"momentum_{n}"for n in [5,10,20]]+[
        "vol_20","price_vs_ma20","vol_vs_ma20",
        "pe_ttm","pb","dv_ttm","roe"
    ]
stock_num=100
future_days_num=5
train_start_date="20150101"
train_end_date="20250721"
pred_start_date="20250101"
pred_end_date="20250721"
my_token="token"
ts.set_token(my_token)
pro=ts.pro_api()
def sort_date(df,list_name):
    df[list_name]=pd.to_datetime(df[list_name])
    return df[list_name]
def sort_values(df,list_name):
    df=df.sort_values(by=list_name)
    return df
def get_stock_list(stock_num):
    print("获取沪深300成分股列表...")
    current_date=datetime.now()
    current_date_str=current_date.strftime("%Y%m%d")
    print(f"今天是{current_date_str}，")
    shift=0
    while True:
        current_date_str=(current_date-timedelta(days=shift)).strftime("%Y%m%d")
        df=pro.index_weight(index_code="000300.SH",trade_date=current_date_str)
        if df.empty:
            print(f"第{shift+1}次尝试，在{current_date_str}没有找到数据，")
            shift+=1
        else:
            print(f"第{shift+1}次尝试，在{current_date_str}处找到了数据。")
            print(df)
            print(f"现在开始抽样，抽样数{stock_num}支，")
            stock_list=df["con_code"].tolist()
            stock_list_sampled=random.sample(stock_list,stock_num)
            print(f"抽样结果，")
            print(stock_list_sampled[:stock_num])
            return stock_list_sampled
def get_known_stock_list(stock_num,date):
    print(f"获取{date}的沪深300成分股列表...")
    df=pro.index_weight(index_code="000300.SH",trade_date=date)
    stock_list=df["con_code"].tolist()
    #stock_list_sampled=random.sample(stock_list,stock_num)
    #print(f"抽样结果，")
    #print(stock_list_sampled[:stock_num])
    print(f"样本数据，")
    print(df)
    return stock_list
raw_code_list=get_known_stock_list(stock_num,"20250701")
#raw_code_list=get_stock_list(stock_num)
def get_dataset(code_list,start_date,end_date):
    all_data=[]
    code_remained=[]
    fina_start_date=str(int(start_date[:4])-2)+start_date[4:]
    counter=1
    for code in code_list:
        df_daily=pro.daily(ts_code=code,start_date=start_date,end_date=end_date)
        df_fina_indicator=pro.fina_indicator(ts_code=code,start_date=fina_start_date,end_date=end_date,
                                             fields="ts_code,ann_date,end_date,roe")
        df_daily_basic=pro.daily_basic(ts_code=code,start_date=start_date,end_date=end_date,
                                       fields="ts_code,trade_date,pb,dv_ttm,pe_ttm")
        print(f"这是第{counter}次获取，")
        if df_daily.empty or df_daily_basic.empty:
            print(f"{code}停牌")
        else:
            df_daily["trade_date"]=sort_date(df_daily,"trade_date")
            df_fina_indicator["ann_date"]=sort_date(df_fina_indicator,"ann_date")
            df_daily_basic["trade_date"]=sort_date(df_daily_basic,"trade_date")
            df_daily=sort_values(df_daily,"trade_date")
            df_fina_indicator = df_fina_indicator.dropna(subset="ann_date")
            df_fina_indicator=sort_values(df_fina_indicator,"ann_date")
            df_daily_basic=sort_values(df_daily_basic,"trade_date")
            df_merged=pd.merge(df_daily,df_daily_basic,on=["ts_code","trade_date"])
            df_final=pd.merge_asof(df_merged,df_fina_indicator,
                                   left_on="trade_date",right_on="ann_date",
                                   by="ts_code",direction="backward")
            all_data.append(df_final)
            code_remained.append(code)
            print(f'成功获取{code}从{start_date}到{end_date}的日线数据和指标，')
            if df_fina_indicator.empty:
                print(f"未能获取{code}从{fina_start_date}到{end_date}的财务数据，")
            else:
                print(f'成功获取{code}从{fina_start_date}到{end_date}的财务数据，')
        time.sleep(2)
        counter+=1
    dataset=pd.concat(all_data,ignore_index=True)
    dataset=dataset.sort_values(by=["ts_code","trade_date"])
    dataset=dataset.set_index("trade_date")
    print(f"完成数据获取，共{len(list(set(code_remained)))}支股票，")
    print(dataset.groupby('ts_code').head(1))
    return [dataset,code_remained]
print(f"获取用于训练的数据，从{train_start_date}到{train_end_date}，")
raw_dataset,code_remained=get_dataset(raw_code_list,train_start_date,train_end_date)
print(f"获取用于预测的数据，从{pred_start_date}到{pred_end_date}，")
pred_material,pred_code_remained=get_dataset(raw_code_list,pred_start_date,pred_end_date)
def get_features(dataset,future_days_num):
    df=dataset.copy()
    df["log_return"]=df.groupby("ts_code")["close"].transform(
        lambda x:np.log(x/x.shift(1))
    )
    for n in [5,10,20]:
        df[f"momentum_{n}"]=df.groupby("ts_code")["log_return"].transform(
            lambda x:x.rolling(n).sum()
        )
    df["vol_20"]=df.groupby("ts_code")["log_return"].transform(
        lambda x:x.rolling(20).std()
    )
    df["price_ma20"]=df.groupby("ts_code")["close"].transform(
        lambda x:x.rolling(20).mean()
    )
    df["vol_ma20"]=df.groupby("ts_code")["vol"].transform(
        lambda x:x.rolling(20).mean()
    )
    df["price_vs_ma20"]=df["close"]/df["price_ma20"]
    df["vol_vs_ma20"]=df["vol"]/df["vol_ma20"]
    df["future_close"]=df.groupby("ts_code")["close"].shift(-future_days_num)
    df["target"]=np.log(df["future_close"]/df["close"])
    df=df.dropna()
    X=df[features]
    y=df["target"]
    code_remained=df["ts_code"]
    print(f"已获取{len(list(set(code_remained)))}支股票的数据，股票列表为：\n"
          f"{code_remained}\n"
          f"特征为，\n"
          f"{X.head(25)}\n"
          f"目标为，\n"
          f"{y.head(25)}\n")
    return [X,y,code_remained]
print(f"处理用于训练的数据，获取特征，以及每日后面{future_days_num}的走势作为因变量，")
X_raw,y_raw,code_remained=get_features(raw_dataset,future_days_num)
print(f"处理用于预测的数据，获取特征，以及每日后面{future_days_num}的走势作为因变量，")
X_raw_material,_,pred_code_list=get_features(pred_material,future_days_num)
print(f"获取用于预测的数据的代码列表，")
X_temp=pd.concat([X_raw_material,pred_code_list],axis=1)
X_temp_sliced=X_temp.groupby("ts_code").tail(1)
pred_feature=X_temp_sliced[features]
pred_code_list=X_temp_sliced["ts_code"]
print(f"成功获取，为，\n"
      f"{pred_code_list}")
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
def test_features_by_XGBRegressor(X,y,n_split=5):
    print(f"在{train_start_date}到{train_end_date}上进行{n_split}分割，"
          f"进行时序交叉模型训练，")
    tss=TimeSeriesSplit(n_splits=n_split)
    losses=[]
    round_num=1
    best_iter=[]
    for train_index,test_index in tss.split(X):
        X_train=X.iloc[train_index]
        X_test=X.iloc[test_index]
        y_train=y.iloc[train_index]
        y_test=y.iloc[test_index]
        print(f"现在是{round_num}/{n_split}折，"
              f"已经划分了训练集和测试集，")
        model = XGBRegressor(n_estimators=1000, learning_rate=0.01,gamma=1,max_depth=3,
                             subsample=0.8,colsample_bytree=0.8,
                             early_stopping_rounds=20, random_state=seed_value)
        scaler = StandardScaler()
        print(f"第{round_num}个模型初始化成功，")
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        model.fit(X_train_scaled,y_train,
                  eval_set=[(X_test_scaled,y_test)],
                  verbose=True)
        print(f"训练完成，")
        pred=model.predict(X_test_scaled)
        mae=mean_absolute_error(y_test,pred)
        iter=model.best_iteration
        print(f"预测完成，预测结果和测试集的差值（mae）为，"
              f"{mae:.4f}，这是第{round_num}次训练结果，"
              f"\n"
              f"这次的最佳迭代次数是{iter}")
        losses.append(mae)
        best_iter.append(iter)
        round_num+=1
    loss_mean=np.mean(losses)
    loss_std=np.std(losses)
    optimal_iter=int(np.mean(best_iter))
    print(f"验证完成，{round_num}次测试中，"
          f"平均损失为{loss_mean:.4f}，"
          f"\n"
          f"标准差为{loss_std:.4f}，"
          f"最终的最佳迭代次数为{optimal_iter}")
    return [loss_mean,loss_std,optimal_iter]
xgb_loss_mean,xgb_loss_std,xgb_optimal_iter=test_features_by_XGBRegressor(X_raw,y_raw)
def apply_XGBRegressor(X,y,iter):
    model=XGBRegressor(n_estimators=iter, learning_rate=0.01,gamma=1,max_depth=3,
                       subsample=0.8,colsample_bytree=0.8,
                       random_state=seed_value)
    pipeline=Pipeline([
        ("scaler",StandardScaler()),
        ("model",model)
    ])
    print("pipeline_xgb 初始化完成，")
    pipeline.fit(X,y)
    print("pipeline_xgb 训练完成，")
    return pipeline
pipeline_xgb=apply_XGBRegressor(X_raw,y_raw,xgb_optimal_iter)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
def test_features_by_keras(X,y,n_split=5):
    print(f"在{train_start_date}到{train_end_date}上进行{n_split}分割，"
          f"进行时序交叉模型训练，")
    tss = TimeSeriesSplit(n_splits=n_split)
    losses = []
    round_num = 1
    n_features=X.shape[1]
    best_epochs=[]
    for train_index, test_index in tss.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        print(f"现在是{round_num}/{n_split}折，"
              f"已经划分了训练集和测试集，")
        print(f"训练集的范围是{X_train.index.min()}到{X_train.index.max()}")
        print(f"测试集的范围是{X_test.index.min()}到{X_test.index.max()}")
        model=keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64,activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32,activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer="adam",loss="mae")
        scaler = StandardScaler()
        print(f"第{round_num}个模型初始化成功，")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        early_stopping=EarlyStopping(monitor="val_loss",
                                     patience=10,
                                     restore_best_weights=True)
        print(f"开始第{round_num}次训练，")
        history=model.fit(X_train_scaled,y_train,
                          validation_data=(X_test_scaled,y_test),
                          epochs=100,
                          batch_size=32,
                          verbose=1,
                          callbacks=[early_stopping]
                          )
        print(f"训练完成，")
        pred = model.predict(X_test_scaled)
        print(pred.shape)
        print(y_test.shape)
        mae = mean_absolute_error(y_test,pred)
        best_epoch=np.argmin(history.history["val_loss"])+1
        print(f"预测完成，预测结果和测试集的差值（mae）为，"
              f"{mae:.4f}，这是第{round_num}次训练结果，"
              f"最佳 epoch 为{best_epoch}"
              )
        best_epochs.append(best_epoch)
        losses.append(mae)
        round_num += 1
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    optimal_epoch=int(np.mean(best_epochs))
    print(f"验证完成，{round_num}次测试中，"
          f"平均损失为{loss_mean:.4f}，"
          f"\n"
          f"标准差为{loss_std:.4f}，"
          f"最终的最佳训练轮次为{optimal_epoch}")
    return [loss_mean, loss_std,optimal_epoch]
keras_loss_mean,keras_loss_std,keras_optimal_epoch=test_features_by_keras(X_raw,y_raw)
print(f"xgb 模型的平均损失{xgb_loss_mean}")
print(f"xgb 模型的标准差{xgb_loss_std}")
print(f"xbg 模型的最优迭代{xgb_optimal_iter}")
print(f"keras 模型的平均损失{keras_loss_mean}")
print(f"keras 模型的标准差{keras_loss_std}")
print(f"keras 模型的最优迭代{keras_optimal_epoch}")
def apply_keras(X,y,epoch):
    n_features=X.shape[1]
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam",loss="mae")
    print("模型初始化完成，")
    regressor = KerasRegressor(model=model,
                               epochs=epoch,
                               batch_size=32,
                               verbose=1)
    print("适配器搭建完成，")
    pipeline=Pipeline([
        ("scaler",StandardScaler()),
        ("model",regressor)
    ])
    print("pipeline_keras 初始化完成，")
    pipeline.fit(X,y)
    print("pipeline_keras 训练完成，")
    return pipeline
pipeline_keras=apply_keras(X_raw,y_raw,keras_optimal_epoch)
print(f"将从{train_start_date}到{train_end_date}中得到的模型应用到从{pred_start_date}到{pred_end_date}期间形成的特征，"
      f"\n"
      f"并且预测未来{future_days_num}天对于{pred_end_date}的收益，")
print(f"使用 pipeline_xgb，")
pred_future_xgb=pipeline_xgb.predict(pred_feature)
print(f"使用 pipeline_keras，\n")
pred_future_keras=pipeline_keras.predict(pred_feature)
print(f"应用结束，\n")
def sort_pred_result(pred_result):
    df=pred_result.copy()
    pred_return=(np.exp(pred_result)-1)*100
    output_result=pd.DataFrame({
        "ts_code":pred_code_list,
        "pred_log_return":pred_result.flatten(),
        "pred_return_%":pred_return.flatten()
    })
    output_result["pred_log_return"]=output_result["pred_log_return"].round(4)
    output_result["pred_return_%"]=output_result["pred_return_%"].round(4)
    return output_result
pred_result_xgb=sort_pred_result(pred_future_xgb)
pred_result_keras=sort_pred_result(pred_future_keras)
print(f"整理完数据了，\n"
      f"xgb 模型的预测结果是，\n"
      f"{pred_result_xgb}\n"
      f"keras 模型的预测结果是,\n"
      f"{pred_result_keras}")
def get_real_data(code_list,start_date,end_date):
    all_data=[]
    for code in code_list:
        df=pro.daily(ts_code=code,start_date=start_date,end_date=end_date)
        if df.empty:
            print(f"{code}今日停牌了，")
        else:
            print(f"成功获取{code}从{start_date}到{end_date}的数据，")
            all_data.append(df)
            time.sleep(2)
    dataset=pd.concat(all_data,ignore_index=True)
    dataset["log_return"]=dataset.groupby("ts_code")["close"].transform(
        lambda x:np.log(x/x.shift(1))
    )
    dataset["return_%"]=dataset.groupby("ts_code")["close"].transform(
        lambda x:(x/x.shift(1)-1)*100
    )
    dataset=dataset.dropna()
    dataset=dataset.sort_values(by="ts_code")
    return dataset
print(f"获取从20250714到20250721的数据，")
real_data_temp=get_real_data(pred_code_list,"20250714","20250721")
real_data=real_data_temp[["ts_code","log_return","return_%"]].copy()
real_data["log_return_sum"]=real_data.groupby("ts_code")["log_return"].transform(
    lambda x:x.rolling(5).sum())
real_data=real_data.dropna()
print(real_data)
compare_data_xgb=pd.merge(real_data.reset_index(),
                      pred_result_xgb.reset_index(),
                      on="ts_code")
#real_mae_xgb=mean_absolute_error(compare_data["log_return"],compare_data_xgb["pred_log_return"])
compare_data_keras=pd.merge(real_data.reset_index(),
                      pred_result_keras.reset_index(),
                      on="ts_code")
print(compare_data_xgb)
print(compare_data_keras)
real_mae_keras=mean_absolute_error(compare_data_keras["log_return_sum"],compare_data_keras["pred_log_return"])
real_mae_xgb=mean_absolute_error(compare_data_xgb["log_return_sum"],compare_data_xgb["pred_log_return"])
print(f"keras 预测收益率与真实收益率的对比：\n"
      f"{compare_data_keras.head(stock_num)}\n"
      f"误差为{real_mae_keras}"
      f"xgb 预测收益率与真实收益率的对比：\n"
      f"{compare_data_xgb.head(stock_num)}\n"
      f"误差为{real_mae_xgb}")








