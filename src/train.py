# 导入必要的包
import sys
import os
sys.path.append('..')

from utils.common import (
    load_and_preprocess_data, prepare_features, create_preprocessor,
    split_data, evaluate_model, plot_roc_curve, save_model
)
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

def train_knn_model(X_train, y_train, preprocessor):
    """
    训练kNN模型
    """
    # 定义超参数网格
    param_grid = {
        'clf__n_neighbors': [3, 5, 9],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan']
    }

    # 创建管道
    knn = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('clf', KNeighborsClassifier())
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    knn_grid = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_train, y_train)

    print(f"KNN最佳参数: {knn_grid.best_params_}, 最佳得分: {knn_grid.best_score_:.4f}")

    return knn_grid.best_estimator_


def train_lr_model(X_train, y_train, preprocessor):
    """
    训练逻辑回归模型
    """
    # 定义超参数网格
    param_grid = {
        'clf__C': [0.03, 0.3, 3, 30],
        'clf__penalty': ['l2'],
        'clf__solver': ['liblinear'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [1000]
    }

    # 创建管道
    lr = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('clf', LogisticRegression())
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    lr_grid = GridSearchCV(lr, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train, y_train.values)

    print(f"逻辑回归最佳参数: {lr_grid.best_params_}, 最佳得分: {lr_grid.best_score_:.4f}")

    return lr_grid.best_estimator_


def train_dt_model(X_train, y_train, preprocessor):
    """
    训练决策树模型
    """
    # 定义超参数网格
    param_grid = {
        'dt__min_samples_split': [4, 10, 20],
        'dt__min_samples_leaf': [2, 4, 8],
        'dt__max_depth': [3, 5, 7, None],
        'dt__criterion': ['gini', 'entropy'],
        'dt__class_weight': [None, 'balanced']
    }

    # 创建管道
    dt = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('dt', DecisionTreeClassifier())
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    dt_grid = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    dt_grid.fit(X_train, y_train)

    print(f"决策树最佳参数: {dt_grid.best_params_}, 最佳得分: {dt_grid.best_score_:.4f}")

    return dt_grid.best_estimator_


def train_rf_model(X_train, y_train, preprocessor):
    """
    训练随机森林模型
    """
    # 定义超参数网格
    param_grid = {
        'clf__n_estimators': [200],
        'clf__max_depth': [4, 6, 8, None],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2],
        'clf__max_features': ['sqrt', 0.7],
        'clf__class_weight': ['balanced', 'balanced_subsample']
    }

    # 创建管道
    rf = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    rf_grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    print(f"随机森林最佳参数: {rf_grid.best_params_}, 最佳得分: {rf_grid.best_score_:.4f}")

    return rf_grid.best_estimator_


def train_ada_model(X_train, y_train, preprocessor):
    """
    训练AdaBoost模型
    """
    # 定义超参数网格
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.5, 1.0],
        'clf__algorithm': ['SAMME']
    }

    # 创建管道
    ada = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('clf', AdaBoostClassifier(random_state=42))
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    ada_grid = GridSearchCV(ada, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    ada_grid.fit(X_train, y_train)

    print(f"AdaBoost最佳参数: {ada_grid.best_params_}, 最佳得分: {ada_grid.best_score_:.4f}")

    return ada_grid.best_estimator_


def train_gbdt_model(X_train, y_train, preprocessor):
    """
    训练梯度提升决策树模型
    """
    # 定义超参数网格
    param_grid = {
        'clf__n_estimators': [100, 150],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_depth': [2, 3, 4],
        'clf__min_samples_split': [2, 4],
        'clf__min_samples_leaf': [1, 2],
        'clf__subsample': [0.8, 1.0]
    }

    # 创建管道
    gbdt = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    gbdt_grid = GridSearchCV(gbdt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    gbdt_grid.fit(X_train, y_train)

    print(f"梯度提升最佳参数: {gbdt_grid.best_params_}, 最佳得分: {gbdt_grid.best_score_:.4f}")

    return gbdt_grid.best_estimator_


def train_xgb_model(X_train, y_train, preprocessor):
    """
    训练XGBoost模型
    """
    # 定义超参数网格
    param_grid = {
        'clf__n_estimators': [100, 150],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_depth': [2, 3, 4],
        'clf__min_child_weight': [1, 3],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0]
    }

    # 创建管道
    xgb_model = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('clf', xgb.XGBClassifier())
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    xgb_grid = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)

    print(f"XGBoost最佳参数: {xgb_grid.best_params_}, 最佳得分: {xgb_grid.best_score_:.4f}")

    return xgb_grid.best_estimator_


def train_stacking_model(X_train, y_train, base_learners, preprocessor):
    """
    训练堆叠模型
    """
    # 元学习器超参数优化
    meta_param_grid = {
        'metal__C': [0.01, 0.1, 1, 10, 100],
        'metal__penalty': ['l1', 'l2'],
        'metal__solver': ['liblinear', 'saga'],
        'metal__class_weight': ['balanced', None]
    }

    # 创建元学习器
    meta_learner = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('metal', LogisticRegression())
    ])

    # 定义交叉验证策略
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索
    meta_grid = GridSearchCV(meta_learner, meta_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    meta_grid.fit(X_train, y_train)

    print(f"元学习器最佳参数: {meta_grid.best_params_}, 最佳得分: {meta_grid.best_score_:.4f}")

    # 提取简单分类器
    final_classifier = meta_grid.best_estimator_.named_steps['metal']

    # 创建堆叠分类器
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=final_classifier,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        stack_method='predict_proba'
    )

    # 训练堆叠模型
    stacking_model.fit(X_train, y_train)

    return stacking_model


def main():
    """
    主函数：训练所有模型
    将结果保存到相应的文件夹中
    """
    try:
        # 创建必要的目录
        os.makedirs('../data/processed', exist_ok=True)
        os.makedirs('../data/results', exist_ok=True)
        os.makedirs('../model', exist_ok=True)

        print("="*60)
        print("开始训练模型流程...")
        print("="*60)

        # 加载和预处理数据
        print("1. 加载和预处理数据...")
        df = load_and_preprocess_data()
        print("数据加载完成")

        # 准备特征和标签
        print("2. 准备特征和标签...")
        X, Y = prepare_features(df)
        print("特征准备完成")

        # 创建预处理器
        print("3. 创建预处理器...")
        preprocessor = create_preprocessor(X)
        print("预处理器创建完成")

        # 划分训练集和测试集
        print("4. 划分训练集和测试集...")
        X_train, X_test, y_train, y_test = split_data(X, Y)
        print(f"数据集划分完成 - 训练集: {len(X_train)}, 测试集: {len(X_test)}")

        # 训练各个模型
        models = []
        model_names = []

        # kNN模型
        print("5. 训练kNN模型...")
        best_knn = train_knn_model(X_train, y_train, preprocessor)
        evaluate_model(best_knn, X_test, y_test)
        plot_roc_curve(best_knn, X_test, y_test, "KNN", save_dir='../data/results')
        save_model(best_knn, "best_knn")
        models.append(best_knn)
        model_names.append('knn')
        print("kNN模型训练完成")

        # 逻辑回归模型
        print("\n6. 训练逻辑回归模型...")
        best_lr = train_lr_model(X_train, y_train, preprocessor)
        evaluate_model(best_lr, X_test, y_test)
        plot_roc_curve(best_lr, X_test, y_test, "Logistic Regression", save_dir='../data/results')
        save_model(best_lr, "best_lr")
        models.append(best_lr)
        model_names.append('lr')
        print("逻辑回归模型训练完成")

        # 决策树模型
        print("\n7. 训练决策树模型...")
        best_dt = train_dt_model(X_train, y_train, preprocessor)
        evaluate_model(best_dt, X_test, y_test)
        plot_roc_curve(best_dt, X_test, y_test, "Decision Tree", save_dir='../data/results')
        save_model(best_dt, "best_dt")
        models.append(best_dt)
        model_names.append('dt')
        print("决策树模型训练完成")

        # 随机森林模型
        print("\n8. 训练随机森林模型...")
        best_rf = train_rf_model(X_train, y_train, preprocessor)
        evaluate_model(best_rf, X_test, y_test)
        plot_roc_curve(best_rf, X_test, y_test, "Random Forest", save_dir='../data/results')
        save_model(best_rf, "best_rf")
        models.append(best_rf)
        model_names.append('rf')
        print("随机森林模型训练完成")

        # AdaBoost模型
        print("\n9. 训练AdaBoost模型...")
        best_ada = train_ada_model(X_train, y_train, preprocessor)
        evaluate_model(best_ada, X_test, y_test)
        plot_roc_curve(best_ada, X_test, y_test, "AdaBoost", save_dir='../data/results')
        save_model(best_ada, "best_ada")
        models.append(best_ada)
        model_names.append('ada')
        print("AdaBoost模型训练完成")

        # 梯度提升决策树模型
        print("\n10. 训练梯度提升决策树模型...")
        best_gbdt = train_gbdt_model(X_train, y_train, preprocessor)
        evaluate_model(best_gbdt, X_test, y_test)
        plot_roc_curve(best_gbdt, X_test, y_test, "Gradient Boosting", save_dir='../data/results')
        save_model(best_gbdt, "best_gbdt")
        models.append(best_gbdt)
        model_names.append('gbdt')
        print("梯度提升决策树模型训练完成")

        # XGBoost模型
        print("\n11. 训练XGBoost模型...")
        best_xgb = train_xgb_model(X_train, y_train, preprocessor)
        evaluate_model(best_xgb, X_test, y_test)
        plot_roc_curve(best_xgb, X_test, y_test, "XGBoost", save_dir='../data/results')
        save_model(best_xgb, "best_xgb")
        models.append(best_xgb)
        model_names.append('xgb')
        print("XGBoost模型训练完成")

        # 训练堆叠模型
        print("\n12. 训练堆叠模型...")
        base_learners = [
            ('knn', best_knn),
            ('lr', best_lr),
            ('dt', best_dt),
            ('rf', best_rf),
            ('ada', best_ada),
            ('gbdt', best_gbdt),
            ('xgb', best_xgb)
        ]

        stacking_model = train_stacking_model(X_train, y_train, base_learners, preprocessor)
        evaluate_model(stacking_model, X_test, y_test)
        plot_roc_curve(stacking_model, X_test, y_test, "Stacking", save_dir='../data/results')
        save_model(stacking_model, "stacking")
        print("堆叠模型训练完成")

        print("\n" + "="*60)
        print("所有模型训练完成！")
        print(f"模型已保存到: {os.path.abspath('../model')}")
        print(f"结果图片已保存到: {os.path.abspath('../data/results')}")
        print(f"处理后的数据已保存到: {os.path.abspath('../data/processed')}")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        return False


# todo 5. 测试代码
if __name__ == '__main__':
    # # 1. 测试: 数据加载和预处理
    # print("1. 测试: 数据加载和预处理...")
    # df = load_and_preprocess_data('../data/train.csv')
    # print(f"    数据加载完成，形状: {df.shape}")

    # # 2. 测试: 特征工程
    # print("\n2. 测试: 特征工程...")
    # X, Y = prepare_features(df)
    # print(f"    特征工程完成，X形状: {X.shape}, Y形状: {Y.shape}")

    # # 3. 测试: 创建预处理器
    # print("\n3. 测试: 创建预处理器...")
    # preprocessor = create_preprocessor(X)
    # print("    预处理器创建完成")

    # # 4. 测试: 数据划分
    # print("\n4. 测试: 数据划分...")
    # X_train, X_test, y_train, y_test = split_data(X, Y)
    # print(f"    数据划分完成 - 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # # 5. 测试: 模型训练（简化版，只训练一个模型）
    # print("\n5. 测试: 模型训练...")
    # print("   训练kNN模型...")
    # best_knn = train_knn_model(X_train, y_train, preprocessor)
    # print("    kNN模型训练完成")

    # # 6. 测试: 模型评估
    # print("\n6. 测试: 模型评估...")
    # acc, f1, auc_score = evaluate_model(best_knn, X_test, y_test)
    # print(f"   模型评估完成 - 准确率: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")

    # # 7. 测试: 模型保存
    # print("\n7. 测试: 模型保存...")
    # save_model(best_knn, "test_knn")
    # print("   模型保存完成")

    # # 8. 测试: ROC曲线绘制
    # print("\n8. 测试: ROC曲线绘制...")
    # plot_roc_curve(best_knn, X_test, y_test, "Test KNN", save_dir='../data/results')
    # print("   ROC曲线绘制完成")

    # print("\n" + "="*60)
    # print("所有测试完成！")
    # print("="*60)
    main()