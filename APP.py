import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
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

# 特征范围定义（包含类型信息用于区分连续/分类特征）
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

# 加载保存的模型和相关组件
@st.cache_resource
def load_assets():
    try:
        # 使用joblib加载资源
        model = joblib.load('best_svm_model.pkl')
        scaler = joblib.load('data_scaler.pkl')
        feature_order = joblib.load('feature_order.pkl')  # 这里加载的是简写列表
        
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
    
    # 检查是否有未定义映射的特征
    missing_mappings = [f for f in feature_order if f not in FEATURE_NAME_MAPPING]
    if missing_mappings:
        st.warning(f"以下特征缺少全称映射: {', '.join(missing_mappings)}")
    
    # 收集用户输入
    input_data = {}
    for feature in feature_order:
        # 获取特征全称，如果没有则使用简写
        feature_fullname = FEATURE_NAME_MAPPING.get(feature, feature)
        info = FEATURE_INFO.get(feature, {})
        
        # 修复：使用与FEATURE_INFO中一致的"numerical"类型判断
        if info.get("type") == "numerical":
            # 连续特征使用滑块，使用定义的default值
            input_data[feature] = st.slider(
                feature_fullname,  # 显示全称
                min_value=info.get("min", 0.0),
                max_value=info.get("max", 100.0),
                value=info.get("default", (info.get("min", 0.0) + info.get("max", 100.0)) / 2),
                format="%.3f"  # 显示三位小数，适应医学数据精度
            )
            
            # 为连续变量添加数值输入选项，方便精确输入
            st.caption(f"精确输入 {feature_fullname} 的值:")
            input_data[feature] = st.number_input(
                label=f"{feature_fullname} (精确值)",
                min_value=info.get("min", 0.0),
                max_value=info.get("max", 100.0),
                value=input_data[feature],
                format="%.3f",
                key=f"{feature}_number"  # 唯一key避免冲突
            )
            
        elif info.get("type") == "categorical":
            # 分类特征使用选择框，并显示友好标签
            options = info.get("options", [0, 1])
            labels = info.get("labels", ["No", "Yes"]) if len(options) == 2 else [str(option) for option in options]
            
            # 使用选择框展示分类选项
            selected_idx = st.selectbox(
                feature_fullname,
                range(len(options)),
                format_func=lambda x: labels[x],
                index=options.index(info.get("default", 0))  # 设置默认值
            )
            input_data[feature] = options[selected_idx]
    
    # 转换为DataFrame并保持特征顺序（使用简写）
    input_df = pd.DataFrame([input_data], columns=feature_order)
    
    # 预测按钮
    if st.button("预测"):
        # 使用归一化器处理输入数据
        input_scaled = scaler.transform(input_df)
        
        # 进行预测
        prediction = model.predict(input_scaled)
        
        # 显示结果
        st.subheader("预测结果")
        st.success(f"预测类别: {prediction[0]}")
        
        # 显示概率（如果模型支持）
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)
            st.write("预测概率:")
            prob_df = pd.DataFrame(
                proba, 
                columns=[f"类别 {i}" for i in range(proba.shape[1])]
            )
            st.dataframe(prob_df.style.format("{:.2%}"))
