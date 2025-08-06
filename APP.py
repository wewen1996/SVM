import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 设置 matplotlib 中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]

# 加载保存的模型
model = joblib.load('svm_model.pkl')

# 特征范围定义
feature_ranges = {
    "BSA": {"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730},
    "Syncope": {"type": "categorical", "options": [0, 1], "default": 0},
    "NtroBNP": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
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

# Streamlit 界面
st.title("PIMSRA围手术期MACE发生概率预测模型")

# 动态生成输入项
st.header("输入以下特征值：")
feature_values = []
feature_display = []  # 用于记录输入特征的展示值
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']}-{properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (选择一个值)",
            options=properties["options"],
        )
    feature_values.append(value)
    feature_display.append(f"{feature}: {value}")  # 记录特征名和对应值

# 转换为模型输入格式
features = np.array([feature_values])
features_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

# 预测与 SHAP 可视化
if st.button("预测"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100
    target_probability = predicted_proba[1] * 100  # 关注事件发生的概率
    
    # 风险分层逻辑（可根据实际需求调整阈值）
    if target_probability < 30:
        risk_level = "低风险"
    elif 30 <= target_probability < 70:
        risk_level = "中风险"
    else:
        risk_level = "高风险"
    
    # 显示输入回顾
    st.subheader("当前输入:")
    st.write(", ".join(feature_display))
    
    # 显示预测结果
    st.subheader("预测结果:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PIMSRA围手术期MACE发生概率", f"{target_probability:.2f}%")
    with col2:
        st.metric("风险等级", risk_level)
    
    # 临床解读
    st.subheader("临床解读:")
    if risk_level == "低风险":
        st.write(" - 低风险 (<30%): 建议常规监测")
    elif risk_level == "中风险":
        st.write(" - 中风险 (30-70%): 建议加强随访观察")
    else:
        st.write(" - 高风险 (≥70%): 建议考虑立即临床干预")
    st.info("注: 本预测工具仅作为临床参考，不能替代医生的专业判断。")
    
    # 计算SHAP值并展示特征重要性
    st.subheader("模型可解释性 (SHAP分析):")
    
    # 创建背景数据（使用默认值作为参考）
    background = pd.DataFrame([[v["default"] for v in feature_ranges.values()]], 
                             columns=feature_ranges.keys())
    
    # 定义预测函数（返回正类概率），适配SHAP解释
    def predict_fn(X):
        return model.predict_proba(X)[:, 1]  # 只关注正类概率
    
    # 初始化解释器（使用KernelExplainer）
    with st.spinner("正在计算特征重要性..."):
        try:
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(features_df)
            
            # 处理 shap_values 类型：若为字典（多分类场景），取关注的类别
            if isinstance(shap_values, dict):
                # 假设关注类别 1，可根据实际情况调整
                shap_values_positive = shap_values.get(1, [])  
            else:
                # 若为列表/数组，直接使用（二分类场景常见）
                shap_values_positive = shap_values if not isinstance(shap_values, list) else shap_values[0]
            
            # 1. 显示单个样本的力导向图
            st.write("### 特征对当前预测的影响:")
            # 处理基准值（explainer.expected_value 可能是字典/数组）
            base_value = explainer.expected_value.get(1, explainer.expected_value) if isinstance(explainer.expected_value, dict) else explainer.expected_value
            
            force_plot = shap.force_plot(
                base_value,  # 基准值（单值或对应类别的值）
                shap_values_positive[0] if hasattr(shap_values_positive, '__getitem__') else shap_values_positive,  
                features_df.iloc[0],          
                matplotlib=False,
                show=False
            )
            
            # 保存为力导向图HTML并显示
            shap.save_html("shap_force_plot.html", force_plot)
            st.components.v1.html(open("shap_force_plot.html", "r").read(), height=200)
        
            
        except Exception as e:
            st.error(f"SHAP分析出错: {str(e)}")


