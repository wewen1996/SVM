import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]

# 加载保存的模型和标签编码器
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_svm_model.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, le
    except FileNotFoundError as e:
        st.error(f"模型文件未找到: {e}")
        return None, None
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None, None

# 加载模型
model, le = load_models()

# 特征范围定义
feature_ranges = {
    "Body Surface Area":{"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730 },
    "History of syncope": {"type": "categorical", "options": [0, 1], "default": 0},
    "N-terminal pro B-type natriuretic peptide": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
    "Hematocrit": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.411},
    "Maximal left ventricular outflow tract gradients": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 102.800},
    "Pulmonary hypertension": {"type": "categorical", "options": [0, 1], "default": 0},
    "Abnormal exercise blood pressure response": {"type": "categorical", "options": [0, 1], "default": 0},
    "History of Septal Reduction Therapy": {"type": "categorical", "options": [0, 1], "default": 0},
    "Usage of Angiotensin Converting Enzyme Inhibitors or Angiotensin Receptor Blockers": {"type": "categorical", "options": [0, 1], "default": 0},
    "Usage of diuretics": {"type": "categorical", "options": [0, 1], "default": 1},
    "Usage of anticoagulant or antiplatelet drugs": {"type": "categorical", "options": [0, 1], "default": 1},
    "Creatinine": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 87.00},
    "Maximum wall thickness": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 37.00},
    "Moderate to severe Systolic Anterior Motion": {"type": "categorical", "options": [0, 1], "default": 1},
    "Diastolic blood pressure": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 95.00},
    "Heart rate": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 84.00},
}

# Streamlit界面
st.title("PIMSRA围手术期MACE发生概率预测模型")

# 如果模型加载成功，显示应用内容
if model and le:
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
                format="%.3f"
            )
        elif properties["type"] == "categorical":
            # 为分类变量添加说明
            options_with_labels = {0: "否", 1: "是"}
            selected_label = st.selectbox(
                label=f"{feature}",
                options=list(options_with_labels.keys()),
                format_func=lambda x: options_with_labels[x],
                index=properties["options"].index(properties["default"])
            )
            value = selected_label
        feature_values.append(value)
        feature_display.append(f"{feature}: {value}")  # 记录特征名和对应值

    # 转换为模型输入格式
    features = np.array([feature_values])
    features_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

    # 预测与SHAP可视化
    if st.button("预测"):
        try:
            # 模型预测
            predicted_class = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]
            
            # 解码预测类别
            predicted_class_label = le.inverse_transform([predicted_class])[0]
            
            # 提取预测的类别概率
            target_probability = predicted_proba[1] * 100  # 关注事件发生的概率
            
            # 风险分层逻辑（可根据实际需求调整阈值）
            if target_probability < 30:
                risk_level = "低风险"
            elif 30 <= target_probability < 70:
                risk_level = "中风险"
            else:
                risk_level = "高风险"
            
            # 显示输入回顾
            with st.expander("查看当前输入特征", expanded=False):
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
                st.write(" - 低风险 (<30%): 建议常规监测，遵循标准临床流程")
            elif risk_level == "中风险":
                st.write(" - 中风险 (30-70%): 建议加强随访观察，考虑优化药物治疗方案")
            else:
                st.write(" - 高风险 (≥70%): 建议考虑立即临床干预，制定详细的风险降低策略")
            st.info("注: 本预测工具仅作为临床参考，不能替代医生的专业判断。")
            
            # 计算SHAP值并展示特征重要性
            st.subheader("模型可解释性 (SHAP分析):")
            
            # 创建背景数据（使用默认值作为参考）
            background = pd.DataFrame(
                [[v["default"] for v in feature_ranges.values()]], 
                columns=feature_ranges.keys()
            )
            
            # 定义预测函数（返回正类概率），适配SHAP解释
            def predict_fn(X):
                return model.predict_proba(X)[:, 1]  # 只关注正类概率
            
            # 初始化解释器（使用KernelExplainer，适合SVM等非树模型）
            with st.spinner("正在计算特征重要性..."):
                try:
                    # 对于SVM，使用KernelExplainer
                    explainer = shap.KernelExplainer(predict_fn, background, link="logit")
                    shap_values = explainer.shap_values(features_df, nsamples=100)
                    
                    # 1. 显示单个样本的力导向图
                    st.write("### 特征对当前预测的影响:")
                    force_plot = shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        features_df.iloc[0],
                        matplotlib=False,
                        show=False,
                        feature_names=feature_ranges.keys()
                    )
                    
                    # 保存为力导向图HTML并显示
                    shap.save_html("shap_force_plot.html", force_plot)
                    st.components.v1.html(open("shap_force_plot.html", "r").read(), height=200)
                    
                    # 2. 显示特征重要性条形图
                    st.write("### 特征重要性排序:")
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(
                        shap_values, 
                        features_df,
                        feature_names=feature_ranges.keys(),
                        plot_type="bar",
                        show=False
                    )
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
                    
                except Exception as e:
                    st.warning(f"SHAP分析出错: {str(e)}。可能需要更多计算资源或调整参数。")
                    st.info("特征重要性分析是可选的，不影响预测结果的准确性。")
        
        except Exception as e:
            st.error(f"预测过程出错: {str(e)}")
else:
    st.error("模型加载失败，请检查模型文件是否存在或完整。")
