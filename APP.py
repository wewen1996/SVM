import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面标题
st.title("PIMSRA围手术期MACE预测模型")

# 设置matplotlib字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        
# 定义变量简写与全称的映射关系
FEATURE_NAME_MAPPING = {
    "BSA": "Body Surface Area" ,
    "Syncope": "History of syncope" ,
    "NTproBNP":"N-terminal pro B-type natriuretic peptide",
    "Hct":"Hematocrit",
    "LVOTGmax": "Maximal left ventricular outflow tract gradients" ,
    "PH": "Pulmonary hypertension",
    "ABPR": "Abnormal exercise blood pressure response",
    "PriorSRT": "History of Septal Reduction Therapy",
    "ACEIARB": "Usage of Angiotensin Converting Enzyme Inhibitors or Angiotensin Receptor Blockers",
    "Diuretics": "Usage of diuretics",
    "AAD":"Usage of anticoagulant or antiplatelet drugs",
    "Cr":"Creatinine",
    "MWT":"Maximum wall thickness",
    "SAMmod":"Moderate to severe Systolic Anterior Motion",
    "DBP": "Diastolic blood pressure",
    "HR": "Heart rate"
}

# 特征范围定义
FEATURE_INFO = {
    "BSA":{"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730 },
    "Syncope": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "NTproBNP": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
    "Hct": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.411},
    "LVOTGmax": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 102.800},
    "PH": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "ABPR": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "PriorSRT": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "ACEIARB": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "Diuretics": {"type": "categorical", "options": [0, 1], "default": 1, "labels": ["No", "Yes"]},
    "AAD": {"type": "categorical", "options": [0, 1], "default": 1, "labels": ["No", "Yes"]},
    "Cr": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 87.00},
    "MWT": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 37.00},
    "SAMmod": {"type": "categorical", "options": [0, 1], "default": 1, "labels": ["No", "Yes"]},
    "DBP": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 95.00},
    "HR": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 84.00},
}

# 加载模型和相关组件
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_svm_model.pkl')
        scaler = joblib.load('data_scaler.pkl')
        feature_order = joblib.load('feature_order.pkl')  # 简写列表
        return model, scaler, feature_order
    except FileNotFoundError as e:
        st.error(f"未找到必要文件: {e.filename}")
        return None, None, None
    except Exception as e:
        st.error(f"加载资源出错: {str(e)}")
        return None, None, None

# 加载资源
model, scaler, feature_order = load_assets()

if model and scaler and feature_order:
    st.subheader("请输入以下信息")
    
    # 检查特征映射是否完整
    missing_mappings = [f for f in feature_order if f not in FEATURE_NAME_MAPPING]
    if missing_mappings:
        st.warning(f"以下特征缺少全称映射: {', '.join(missing_mappings)}")
    
    # 收集用户输入
    input_data = {}
    for feature in feature_order:
        feature_fullname = FEATURE_NAME_MAPPING.get(feature, feature)
        info = FEATURE_INFO.get(feature, {})
        
        if info.get("type") == "numerical":
            # 连续特征：滑块+精确输入
            input_data[feature] = st.slider(
                feature_fullname,
                min_value=info.get("min", 0.0),
                max_value=info.get("max", 100.0),
                value=info.get("default", (info.get("min", 0.0) + info.get("max", 100.0)) / 2),
                format="%.3f"
            )
            input_data[feature] = st.number_input(
                label=f"{feature_fullname} (精确值)",
                min_value=info.get("min", 0.0),
                max_value=info.get("max", 100.0),
                value=input_data[feature],
                format="%.3f",
                key=f"{feature}_number"
            )
        elif info.get("type") == "categorical":
            # 分类特征：选择框
            options = info.get("options", [0, 1])
            labels = info.get("labels", ["No", "Yes"]) if len(options) == 2 else [str(option) for option in options]
            selected_idx = st.selectbox(
                feature_fullname,
                range(len(options)),
                format_func=lambda x: labels[x],
                index=options.index(info.get("default", 0))
            )
            input_data[feature] = options[selected_idx]
    
    # 关键修改：将input_df的列名从“简写”改为“全称”（与scaler匹配）
    # 1. 生成全称列表（与feature_order顺序一致）
    feature_fullnames = [FEATURE_NAME_MAPPING[feat] for feat in feature_order]
    # 2. 创建input_df时使用全称作为列名
    input_df = pd.DataFrame([input_data], columns=feature_fullnames)
    
    # 预测按钮
    if st.button("预测"):
        # 调试：显示特征名称匹配情况
        st.subheader("特征名称校验（调试信息）")
        if hasattr(scaler, 'feature_names_in_'):
            st.write("scaler预期的特征名称：", list(scaler.feature_names_in_))
            st.write("输入数据的特征名称：", input_df.columns.tolist())
            
            # 检查是否完全匹配
            if list(input_df.columns) == list(scaler.feature_names_in_):
                st.success("特征名称完全匹配！")
            else:
                st.error("特征名称不匹配，请检查上述列表")
                st.stop()
        
        # 归一化和预测
        try:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            
            st.subheader("预测结果")
            st.success(f"预测类别: {prediction[0]}")
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)
                st.write("预测概率:")
                prob_df = pd.DataFrame(
                    proba, 
                    columns=[f"类别 {i}" for i in range(proba.shape[1])]
                )
                st.dataframe(prob_df.style.format("{:.2%}"))
        except Exception as e:
            st.error(f"预测过程出错：{str(e)}")
