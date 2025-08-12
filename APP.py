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

# 定义变量简写与全称的映射关系（界面显示用）
FEATURE_NAME_MAPPING = {
    "BSA": "Body Surface Area",
    "Syncope": "History of syncope",
    "NTproBNP": "N-terminal pro B-type natriuretic peptide",
    "Hct": "Hematocrit",
    "LVOTGmax": "Maximal left ventricular outflow tract gradients",
    "PH": "Pulmonary hypertension",
    "ABPR": "Abnormal exercise blood pressure response",
    "PriorSRT": "History of Septal Reduction Therapy",
    "ACEIARB": "Usage of Angiotensin Converting Enzyme Inhibitors or Angiotensin Receptor Blockers",
    "Diuretics": "Usage of diuretics",
    "AAD": "Usage of anticoagulant or antiplatelet drugs",
    "Cr": "Creatinine",
    "MWT": "Maximum wall thickness",
    "SAMmod": "Moderate to severe Systolic Anterior Motion",
    "DBP": "Diastolic blood pressure",
    "HR": "Heart rate"
}

# 特征范围定义（包含类型信息）
FEATURE_INFO = {
    "BSA": {"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730},
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
    
    # 1. 显式获取模型训练时的特征名称（关键！）
    model_features = []
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        st.write("模型训练时的特征名称：", model_features)  # 调试用
    else:
        st.warning("模型不支持查看训练时的特征名称（可能是旧版本sklearn）")
    
    # 2. 检查feature_order与模型特征的匹配性
    if model_features:
        # 检查是否有多余特征
        extra_features = [f for f in feature_order if f not in model_features]
        if extra_features:
            st.error(f"输入包含模型未见过的特征：{extra_features}")
        
        # 检查是否有缺失特征
        missing_features = [f for f in model_features if f not in feature_order]
        if missing_features:
            st.error(f"输入缺少模型必需的特征：{missing_features}")
        
        # 检查顺序是否一致
        if feature_order != model_features:
            st.warning("特征顺序与模型训练时不一致，可能导致预测错误！")
            st.warning(f"模型预期顺序：{model_features}")
            st.warning(f"当前输入顺序：{feature_order}")
    
    # 3. 收集用户输入（仅包含模型训练时有的特征）
    input_data = {}
    # 只保留模型训练时有的特征（过滤掉多余特征）
    valid_features = [f for f in feature_order if f in model_features] if model_features else feature_order
    
    for feature in valid_features:
        feature_fullname = FEATURE_NAME_MAPPING.get(feature, feature)
        info = FEATURE_INFO.get(feature, {})
        
        if info.get("type") == "numerical":
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
            options = info.get("options", [0, 1])
            labels = info.get("labels", ["No", "Yes"]) if len(options) == 2 else [str(option) for option in options]
            selected_idx = st.selectbox(
                feature_fullname,
                range(len(options)),
                format_func=lambda x: labels[x],
                index=options.index(info.get("default", 0)),
                key=f"{feature}_select"
            )
            input_data[feature] = options[selected_idx]
    
    # 4. 构建输入DataFrame（确保特征顺序与模型完全一致）
    # 关键：用模型训练时的特征顺序，而非feature_order
    input_columns = model_features if model_features else valid_features
    input_df = pd.DataFrame([input_data], columns=input_columns)
    
    # 5. 预测按钮
    if st.button("预测"):
        try:
            # 归一化
            input_scaled = scaler.transform(input_df)
            # 预测
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
            # 显示详细调试信息
            st.subheader("调试信息")
            st.write("输入数据列名：", input_df.columns.tolist())
            st.write("输入数据形状：", input_df.shape)
            if model_features:
                st.write("模型预期列名：", model_features)
                st.write("模型预期特征数量：", len(model_features))
