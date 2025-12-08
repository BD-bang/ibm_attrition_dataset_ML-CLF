# 导入第三方包
import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path='../data/train.csv'):
    """
    加载和预处理数据
    """
    # 加载训练数据
    train_data = pd.read_csv(data_path)
    df = pd.DataFrame(train_data)

    # 去除冗余特征
    df.drop(columns=['Over18', 'StandardHours', 'EmployeeNumber'], inplace=True)

    # 保存原始数据集到data/processed文件夹
    os.makedirs('../data/processed', exist_ok=True)
    df_original = df.copy()
    df_original.to_pickle('../data/processed/df_original.pkl')

    return df


def prepare_features(df):
    """
    准备特征和标签
    """
    # 标签列
    Y = df['Attrition']

    # 使用LabelEncoder转换标签数据
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = pd.DataFrame(Y, columns=['Attrition'])

    # 特征列
    X = df.drop(columns=df.columns[0])

    # 转换特征数据
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = le.fit_transform(X[col])

    # 选取的特征
    choose_feature = ['Age', 'DistanceFromHome','OverTime','Department',
                                'MonthlyIncome', 'NumCompaniesWorked', 'EnvironmentSatisfaction',
                                'StockOptionLevel', 'TotalWorkingYears']

    # 与选取的特征取交集
    X = X[X.columns.intersection(choose_feature).tolist()]

    # 使用ADASYN进行样本均衡
    adasyn = ADASYN(random_state=42)
    X, Y = adasyn.fit_resample(X, Y)

    # 保存处理后的数据到data/processed文件夹
    os.makedirs('../data/processed', exist_ok=True)
    Y.to_pickle('../data/processed/Y.pkl')
    X.to_pickle('../data/processed/X.pkl')

    return X, Y


def create_preprocessor(X):
    """
    创建数据预处理器
    """
    # 无序字符型
    nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    # 有序字符型
    ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                   'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
                   'StockOptionLevel', 'WorkLifeBalance']

    # 数字连续型
    continuous_cols = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
                      'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
                      'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    # 获取实际存在的特征
    choose_feature = X.columns.tolist()
    nominal_cols = list(set(nominal_cols) & set(choose_feature))
    ordinal_cols = list(set(ordinal_cols) & set(choose_feature))
    continuous_cols = list(set(continuous_cols) & set(choose_feature))

    # 创建预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols),
            ('ord', OrdinalEncoder(), ordinal_cols),
            ('cont', StandardScaler(), continuous_cols)
        ],
    ).set_output(transform="pandas")

    return preprocessor


def split_data(X, Y, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    """
    # 使用SMOTE进行样本均衡
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(X, Y)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=True, stratify=Y)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    评估模型性能
    """
    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 预测概率
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 打印结果
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {roc_auc:.4f}")
    print(f"混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print(f"分类报告:")
    print(classification_report(y_test, y_pred))

    return acc, f1, roc_auc


def plot_roc_curve(model, X_test, y_test, model_name="Model", save_dir='../data/results'):
    """
    绘制ROC曲线
    """
    # 预测概率
    y_score = model.predict_proba(X_test)[:, 1]

    # 计算FPR、TPR、AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 创建结果保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘图
    from sklearn.metrics import RocCurveDisplay
    plt.figure(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                   estimator_name=model_name).plot(color='darkorange', lw=2)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'{save_dir}/roc_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc


def save_model(model, model_name, save_dir="../model"):
    """
    保存模型
    """
    os.makedirs(save_dir, exist_ok=True)
    dump(model, os.path.join(save_dir, f"{model_name}.pkl"))
    print("模型已保存到:", os.path.abspath(save_dir))


# todo 5. 测试代码
if __name__ == '__main__':
    # 1. 测试: 数据加载和预处理
    print("1. 测试: 数据加载和预处理...")
    df = load_and_preprocess_data('../data/train.csv')
    print(f"   数据加载完成，形状: {df.shape}")

    # 2. 测试: 特征工程
    print("\n2. 测试: 特征工程...")
    X, Y = prepare_features(df)
    print(f"   特征工程完成，X形状: {X.shape}, Y形状: {Y.shape}")

    # 3. 测试: 创建预处理器
    print("\n3. 测试: 创建预处理器...")
    preprocessor = create_preprocessor(X)
    print("   预处理器创建完成")

    # 4. 测试: 数据划分
    print("\n4. 测试: 数据划分...")
    X_train, X_test, y_train, y_test = split_data(X, Y)
    print(f"   数据划分完成 - 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 5. 测试: 模型评估功能
    print("\n5. 测试: 模型评估功能...")
    # 创建一个简单的虚拟模型用于测试
    from sklearn.dummy import DummyClassifier
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(X_train, y_train)
    acc, f1, auc_score = evaluate_model(dummy_model, X_test, y_test)
    print(f"   模型评估完成 - 准确率: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")

    # 6. 测试: ROC曲线绘制
    print("\n6. 测试: ROC曲线绘制...")
    plot_roc_curve(dummy_model, X_test, y_test, "Test Dummy Model", save_dir='../data/results')
    print("   ROC曲线绘制完成")

    # 7. 测试: 模型保存
    print("\n7. 测试: 模型保存...")
    save_model(dummy_model, "test_dummy_model")
    print("   模型保存完成")

    print("\n" + "="*60)
    print("common.py 所有测试完成！")
    print("="*60)