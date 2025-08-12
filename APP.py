import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面标题
st.title("PIMSRA围手术期MACE预测模型")

# 设置matplotlib字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 核心映射：简写（实际用）→ 全称（显示用）
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

# 2. 特征元信息（基于简写定义）
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

# 3. 加载模型和核心特征信息
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_svm_model.pkl')
        scaler = joblib.load('data_scaler.pkl')
        
        # 从模型中提取训练时的特征名称（简称）和顺序（最权威的来源）
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            # 若模型不支持feature_names_in_，则从feature_order.pkl加载（确保与训练一致）
            model_feature_names = joblib.load('feature_order.pkl')
        
        return model, scaler, model_feature_names
    except FileNotFoundError as e:
        st.error(f"未找到文件: {e.filename}")
        return None, None, None
    except Exception as e:
        st.error(f"加载资源失败: {str(e)}")
        return None, None, None

# 4. 加载资源
model, scaler, model_feature_names = load_assets()

if model and scaler and model_feature_names:
    st.subheader("请输入以下信息")
    
    # 显示模型训练时的特征（供调试，确认是简称）
    with st.expander("点击查看模型训练时的特征（调试用）"):
        st.write("模型预期的特征名称（简称）：", model_feature_names)
    
    # 5. 收集用户输入（键为简写，显示为全称）
    input_data = {}
    for feature in model_feature_names:  # 严格按模型特征顺序遍历
        # 确保特征在映射中存在
        if feature not in FEATURE_NAME_MAPPING:
            st.warning(f"特征 {feature} 缺少全称映射，将显示简写")
            feature_fullname = feature
        else:
            feature_fullname = FEATURE_NAME_MAPPING[feature]
        
        info = FEATURE_INFO.get(feature, {})
        if not info:
            st.error(f"特征 {feature} 缺少元信息（类型/范围）")
            continue
        
        # 连续特征：滑块+精确输入
        if info["type"] == "numerical":
            input_data[feature] = st.slider(
                label=feature_fullname,
                min_value=info["min"],
                max_value=info["max"],
                value=info["default"],
                format="%.3f",
                key=f"{feature}_slider"
            )
            input_data[feature] = st.number_input(
                label=f"{feature_fullname}（精确值）",
                min_value=info["min"],
                max_value=info["max"],
                value=input_data[feature],
                format="%.3f",
                key=f"{feature}_num"
            )
        
        # 分类特征：选择框
        elif info["type"] == "categorical":
            options = info["options"]
            labels = info.get("labels", [str(opt) for opt in options])
            selected_idx = st.selectbox(
                label=feature_fullname,
                options=range(len(options)),
                format_func=lambda i: labels[i],
                index=options.index(info["default"]),
                key=f"{feature}_cat"
            )
            input_data[feature] = options[selected_idx]
    
    # 6. 构建输入数据（列名=模型特征名称，顺序完全一致）
    input_df = pd.DataFrame([input_data], columns=model_feature_names)
    
    # 7. 预测逻辑
    if st.button("预测"):
        try:
            # 归一化（使用训练好的scaler）
            input_scaled = scaler.transform(input_df)
            
            # 模型预测
            prediction = model.predict(input_scaled)
            st.subheader("预测结果")
            st.success(f"预测类别: {prediction[0]}")
            
            # 显示概率（若支持）
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)
                st.write("预测概率:")
                prob_df = pd.DataFrame(
                    proba,
                    columns=[f"类别 {i}" for i in range(proba.shape[1])]
                )
                st.dataframe(prob_df.style.format("{:.2%}"))
        
        except Exception as e:
            st.error(f"预测出错: {str(e)}")
            # 详细调试信息
            with st.expander("查看详细调试信息"):
                st.write("输入数据列名（简称）：", input_df.columns.tolist())
                st.write("输入数据形状：", input_df.shape)
                st.write("模型预期特征数量：", len(model_feature_names))
                if hasattr(scaler, 'feature_names_in_'):
                    st.write("Scaler预期特征名称：", list(scaler.feature_names_in_))
