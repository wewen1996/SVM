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
        
#  定义变量简写与全称的映射关系
FEATURE_NAME_MAPPING = {
    "Body Surface Area": "BSA",
    "History of syncope": "Syncope",
    "N-terminal pro B-type natriuretic peptide": "NTproBNP",
    "Hematocrit": "Hct",
    "Maximal left ventricular outflow tract gradients": "LVOTGmax",
    "Pulmonary hypertension": "PH",
    "Abnormal exercise blood pressure response": "ABPR",
    "History of Septal Reduction Therapy": "PriorSRT",
    "Usage of Angiotensin Converting Enzyme Inhibitors or Angiotensin Receptor Blockers": "ACEIARB",
    "Usage of diuretics": "Diuretics",
    "Usage of anticoagulant or antiplatelet drugs": "AAD",
    "Creatinine": "Cr",
    "Maximum wall thickness": "MWT",
    "Moderate to severe Systolic Anterior Motion": "SAMmod",
    "Diastolic blood pressure": "DBP",
    "Heart rate": "HR"
}

# 特征范围定义（包含类型信息用于区分连续/分类特征）
FEATURE_INFO = {
    "BSA":{"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730 },
    "Syncope": {"type": "categorical", "options": [0, 1], "default": 0},
    "NTproBNP": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
    "Hct": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.411},
    "LVOTGmax": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 102.800},
    "PH": {"type": "categorical", "options": [0, 1], "default": 0},
    "ABPR": {"type": "categorical", "options": [0, 1], "default": 0},
    "PriorSRT": {"type": "categorical", "options": [0, 1], "default": 0},
    "ACEIARB": {"type": "categorical", "options": [0, 1], "default": 0},
    "Diuretics": {"type": "categorical", "options": [0, 1], "default": 1},
    "AAD": {"type": "categorical", "options": [0, 1], "default": 1},
    "Cr": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 87.00},
    "MWT": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 37.00},
    "SAMmod": {"type": "categorical", "options": [0, 1], "default": 1},
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
        
        if info.get("type") == "continuous":
            # 连续特征使用滑块
            input_data[feature] = st.slider(
                feature_fullname,  # 显示全称
                min_value=info.get("min", 0),
                max_value=info.get("max", 100),
                value=(info.get("min", 0) + info.get("max", 100)) / 2
            )
        else:
            # 分类特征使用单选按钮或选择框
            if info.get("type") == "binary":
                # 二分类特征
                input_data[feature] = st.radio(
                    feature_fullname,  # 显示全称
                    options=info.get("options", [0, 1]),
                    format_func=lambda x: "是" if x == 1 else "否" if len(info.get("options", [])) == 2 else str(x)
                )
            else:
                # 多分类特征
                input_data[feature] = st.selectbox(
                    feature_fullname,  # 显示全称
                    options=info.get("options", [])
                )
    
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








