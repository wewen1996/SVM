
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


## 加载保存的随机森林模型


model = joblib.load('svm_model.pkl')


## 特征范围定义（根据提供的特征范围和数据类型）


feature_ranges = {
    "BSA":{"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730 },
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


## Streamlit 界面


st.title("Prediction Model with SHAP Visualization")


## 动态生成输入项


st.header("Enter the following feature values:")
feature_values = []
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
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)


## 转换为模型输入格式


features = np.array([feature_values])


## 预测与 SHAP 可视化


if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of MACE occurrence in PIMSRA is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    plt.close(fig)  # 关闭当前fig
    st.image("prediction_text.png")

    # 计算 SHAP 值
    background = pd.DataFrame([[v["default"] for v in feature_ranges.values()]], columns=feature_ranges.keys())
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    
    # 生成 SHAP 力图
    class_index = predicted_class
    class_index = max(0, min(1, class_index))  # 强制二分类索引范围
    
    # 调整shap_values访问方式
    shap_values_for_class = shap_values[class_index] if len(shap_values) > class_index else shap_values[0]
    
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index] if len(explainer.expected_value) > class_index else explainer.expected_value,
        shap_values_for_class,
        pd.DataFrame([feature_values], columns=feature_ranges.keys())
    )
    
    # 保存并显示 SHAP 图
    shap.save_html("shap_force_plot.html", shap_fig)  # 先保存为HTML
    # 使用kaleido将HTML转换为PNG
    from kaleido.scopes import PlotlyScope
    scope = PlotlyScope()
    with open("shap_force_plot.png", "wb") as f:
        f.write(scope.transform(shap_fig))
    
    st.image("shap_force_plot.png")  # 显示图像



