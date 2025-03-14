import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 标题文本
title = "Prediction of settlement of high fill canal levee based on Bayesian-XGBoost explicable model"

st.set_page_config(    
    page_title=f"{title}",
    page_icon="⭕",
    layout="wide"
)

# 导入模型
with open("xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# 导入便准化
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

importances = xgb_model.feature_importances_
names = ['ms', 'jf', 'dz', 'tm', 'psb', 'njl', 'mcj', 'node']

st.markdown(f'''
    <h1 style="font-size: 20px; color: white; text-align: center; background: #008BFB; border-radius: .5rem; margin-bottom: 1rem;">
    {title}
    </h1>''', unsafe_allow_html=True)
    
st.markdown('''
    <style>
    .stImage {
        border: 1px solid gray;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    </style>''', unsafe_allow_html=True)

inputdata = {}

expander = st.expander("**Predict input**", True)
with expander:
    columns = st.columns(4)
    
    inputdata["ms"] = columns[0].selectbox("Groundwater depth", [2, 5.5, 9])
    inputdata["jf"] = columns[1].selectbox("Groundwater level reduction", [3, 5.5, 8])
    inputdata["dz"] = columns[2].selectbox("Height of mound", [1, 7, 13])
    inputdata["tm"] = columns[3].selectbox("Elastic modulus", [16, 20, 24])
    inputdata["psb"] = columns[0].selectbox("Poisson's ratio", [0.25, 0.335, 0.42])
    inputdata["njl"] = columns[1].selectbox("Cohesion force", [16, 20, 24])
    inputdata["mcj"] = columns[2].selectbox("Internal friction Angle", [16, 19, 22])
    inputdata["node"] = columns[3].selectbox("Location", [1, 2, 3, 4, 5, 6, 7, 8])
    
    st.info("The parameter values are suggested to refer to the approximate values in the engineering geological exploration report and the field investigation results.")
    
predata = pd.DataFrame([inputdata])[names]
data = predata.copy()
data = scaler.fit_transform(data)

with expander:
    layoutc = st.columns([1, 2, 1])
    # 位置图片匹配，位置图片放在Picture文件夹里面，命名为node_*.png，*为node编号
    layoutc[1].image(f"Picture/node_{int(inputdata['node'])}.tif", use_column_width=True, caption="Point location diagram of predicted settlement value")

with st.expander("**Predict result**", True):
    res = xgb_model.predict(data).flatten()[0]
    
    st.markdown(f'''
        <div style="font-size: 25px; font-weight: bold; text-align: center; color: red; background: transparent; border-bottom: 1px solid black;">
        Predict result is: {round(float(res), 2)}
        </div><br>''', unsafe_allow_html=True)
    
    # 使用 SHAP 解释模型
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb(data)
    
    # 生成 XGBoost 模型预测的第一个样本的 SHAP 值力图
    shap.plots.force(shap_values_xgb[0], matplotlib=True, show=False, features=predata, feature_names=names)
    plt.tick_params(axis='x')
    plt.title('SHAP Force Plot for XGBoost')
    
    layoutc = st.columns([1, 4, 1])
    layoutc[1].pyplot(plt.gcf(), use_container_width=True)

    # 生成 XGBoost 模型预测瀑布图
    exp = shap.Explanation(
                        shap_values_xgb[0].values, 
                        shap_values_xgb[0].base_values, 
                        data=predata.iloc[0].values, 
                        feature_names=names)
    fig = plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.title('SHAP Waterfall Plot for XGBoost')
    
    layoutc = st.columns([1.5, 2, 1.5])
    layoutc[1].pyplot(plt.gcf(), use_container_width=True)









