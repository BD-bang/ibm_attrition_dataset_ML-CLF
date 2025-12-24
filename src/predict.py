# 导入必要的包
import os
import sys
sys.path.append('..')

import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# 导入日志工具
sys.path.append('..')
from utils.log import Logger

import warnings
warnings.filterwarnings('ignore')

def load_test_data(test_path='../data/test2.csv'):
    """
    加载测试数据
    """
    # 加载测试数据
    test_data = pd.read_csv(test_path)
    df_test = pd.DataFrame(test_data)

    # 分离特征和标签
    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']

    return X_test, y_test


def load_model(model_name, model_dir='../model'):
    """
    加载模型
    """
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    model = load(model_path)
    return model


def predict_row_by_row(model, test_data_path, logger):
    """
    对test2.csv进行逐行预测并记录日志
    """
    try:
        # 加载测试数据
        test_data = pd.read_csv(test_data_path)
        logger.info(f"开始逐行预测文件: {test_data_path}")
        logger.info(f"测试数据共有 {len(test_data)} 行")

        predictions = []
        probabilities = []

        # 逐行预测
        for index, row in test_data.iterrows():
            try:
                # 准备单行数据（需要排除标签列）
                row_dict = row.to_dict()
                if 'Attrition' in row_dict:
                    row_dict.pop('Attrition')

                # 转换为DataFrame
                row_df = pd.DataFrame([row_dict])

                # 预测
                pred = model.predict(row_df)[0]
                prob = model.predict_proba(row_df)[0][1]  # 获取正类的概率

                predictions.append(pred)
                probabilities.append(prob)

                # 记录日志
                row_num = index + 1
                logger.info(f"行 {row_num}: 预测={pred}, 概率={prob:.4f}")

                if row_num % 100 == 0:  # 每100行记录一次进度
                    logger.info(f"已处理 {row_num} 行")

            except Exception as e:
                row_num = index + 1
                logger.error(f"行 {row_num} 预测失败: {str(e)}")
                predictions.append(None)
                probabilities.append(None)

        # 添加预测结果到原始数据
        test_data['Prediction'] = predictions
        test_data['Probability'] = probabilities

        # 保存预测结果
        result_path = test_data_path.replace('.csv', '_predictions.csv')
        test_data.to_csv(result_path, index=False)
        logger.info(f"逐行预测完成，结果已保存到: {result_path}")

        return test_data

    except Exception as e:
        logger.error(f"逐行预测过程失败: {str(e)}")
        return None


def predict_and_evaluate(model, X_test, y_test, model_name="Model"):
    """
    进行预测并评估模型
    """
    # 进行预测
    y_pred = model.predict(X_test)

    # 预测概率
    y_score = model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_score)

    # 打印结果
    print(f"模型: {model_name}")
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    return accuracy, f1, auc_score


def plot_roc_curve(model, X_test, y_test, model_name="Model", save_dir='../data/results'):
    """
    绘制ROC曲线
    """
    # 预测概率
    y_score = model.predict_proba(X_test)[:, 1]

    # 计算FPR、TPR、AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    # 创建结果保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘图
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
    plt.savefig(f'{save_dir}/roc_curve_{model_name.lower().replace(" ", "_")}_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"测试集 AUC: {roc_auc:.4f}")

    return roc_auc


def main():
    """
    主函数：加载模型并进行预测
    将预测结果保存到相应的文件夹中，并对test2.csv进行逐行预测
    """
    # 创建日志记录器
    logger = Logger('../', 'predictions').get_logger()
    logger.info("="*60)
    logger.info("开始模型预测评估流程...")
    logger.info("="*60)

    try:
        # 创建必要的目录
        os.makedirs('../data/results', exist_ok=True)

        print("="*60)
        print("开始模型预测评估流程...")
        print("日志记录已启动...")
        print("="*60)

        # 加载测试数据
        print("1. 加载测试数据...")
        logger.info("开始加载测试数据...")
        X_test, y_test = load_test_data()
        print(f"测试数据加载完成 - 样本数: {len(X_test)}")
        logger.info(f"测试数据加载完成 - 样本数: {len(X_test)}")

        # 获取模型列表
        model_dir = '../model'
        if not os.path.exists(model_dir):
            print(f"模型目录不存在: {model_dir}")
            logger.error(f"模型目录不存在: {model_dir}")
            return False

        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

        if not model_files:
            print(f"未找到模型文件在: {model_dir}")
            logger.error(f"未找到模型文件在: {model_dir}")
            return False

        print(f"\n找到 {len(model_files)} 个模型文件:")
        logger.info(f"找到 {len(model_files)} 个模型文件")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")
            logger.info(f"模型文件 {i}: {model_file}")

        # 对每个模型进行预测和评估
        results_summary = []
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            print(f"\n{'='*50}")
            print(f"评估模型: {model_name}")
            print(f"{'='*50}")
            logger.info(f"开始评估模型: {model_name}")

            try:
                # 加载模型
                model = load_model(model_name)
                print(f"模型 {model_name} 加载成功")
                logger.info(f"模型 {model_name} 加载成功")

                # 预测和评估
                accuracy, f1, auc_score = predict_and_evaluate(model, X_test, y_test, model_name)
                results_summary.append({
                    'model': model_name,
                    'accuracy': accuracy,
                    'f1': f1,
                    'auc': auc_score
                })
                logger.info(f"模型 {model_name} 评估结果 - 准确率: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")

                # 绘制ROC曲线
                plot_roc_curve(model, X_test, y_test, model_name, save_dir='../data/results')
                print(f"模型 {model_name} 评估完成")
                logger.info(f"模型 {model_name} 评估完成")

                # 对test2.csv进行逐行预测
                test2_path = '../data/test2.csv'
                if os.path.exists(test2_path):
                    print("开始对 test2.csv 进行逐行预测...")
                    predictions_df = predict_row_by_row(model, test2_path, logger)
                    if predictions_df is not None:
                        print("test2.csv 逐行预测完成")
                        logger.info("test2.csv 逐行预测完成")
                else:
                    logger.warning(f"test2.csv 文件不存在: {test2_path}")

            except Exception as e:
                print(f"评估模型 {model_name} 时出错: {str(e)}")
                logger.error(f"评估模型 {model_name} 时出错: {str(e)}")
                continue

        # 打印结果汇总
        if results_summary:
            print("\n" + "="*60)
            print("模型评估结果汇总:")
            print("="*60)
            print(f"{'模型':<20} {'准确率':<10} {'F1分数':<10} {'AUC':<10}")
            print("-"*50)

            logger.info("模型评估结果汇总:")
            logger.info("-"*50)
            for result in results_summary:
                print(f"{result['model']:<20} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['auc']:<10.4f}")
                logger.info(f"模型 {result['model']}: 准确率={result['accuracy']:.4f}, F1={result['f1']:.4f}, AUC={result['auc']:.4f}")

        print("\n预测评估完成!")
        print(f"结果图片已保存到: {os.path.abspath('../data/results')}")
        print("="*60)

        logger.info("预测评估完成！")
        logger.info(f"结果图片已保存到: {os.path.abspath('../data/results')}")
        logger.info("="*60)

        return True

    except Exception as e:
        print(f"\n预测过程中出现错误: {str(e)}")
        logger.error(f"预测过程中出现错误: {str(e)}")
        return False


# todo 5. 测试代码
if __name__ == '__main__':
    # # 1. 测试: 日志记录器初始化
    # print("1. 测试: 日志记录器初始化...")
    # logger = Logger('../', 'test_predictions').get_logger()
    # logger.info("开始预测测试...")
    # print("日志记录器初始化完成")

    # # 2. 测试: 测试数据加载
    # print("\n2. 测试: 测试数据加载...")
    # try:
    #     X_test, y_test = load_test_data('../data/test2.csv')
    #     print(f"测试数据加载完成 - 样本数: {len(X_test)}")
    #     logger.info(f"测试数据加载完成 - 样本数: {len(X_test)}")
    # except Exception as e:
    #     print(f"测试数据加载失败: {str(e)}")
    #     logger.error(f"测试数据加载失败: {str(e)}")

    # # 3. 测试: 模型加载
    # print("\n3. 测试: 模型加载...")
    # try:
    #     # 尝试加载一个测试模型
    #     model_files = [f for f in os.listdir('../model') if f.endswith('.pkl')]
    #     if model_files:
    #         test_model_name = model_files[0].replace('.pkl', '')
    #         model = load_model(test_model_name)
    #         print(f"模型 {test_model_name} 加载成功")
    #         logger.info(f"模型 {test_model_name} 加载成功")
    #     else:
    #         print("未找到模型文件，跳过模型加载测试")
    #         logger.warning("未找到模型文件，跳过模型加载测试")
    # except Exception as e:
    #     print(f"模型加载失败: {str(e)}")
    #     logger.error(f"模型加载失败: {str(e)}")

    # # 4. 测试: 模型预测和评估
    # print("\n4. 测试: 模型预测和评估...")
    # try:
    #     if 'model' in locals() and 'X_test' in locals() and 'y_test' in locals():
    #         accuracy, f1, auc_score = predict_and_evaluate(model, X_test, y_test, "Test Model")
    #         print(f"模型评估完成 - 准确率: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    #         logger.info(f"模型评估结果 - 准确率: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    #     else:
    #         print("缺少必要数据，跳过预测评估测试")
    #         logger.warning("缺少必要数据，跳过预测评估测试")
    # except Exception as e:
    #     print(f"模型预测评估失败: {str(e)}")
    #     logger.error(f"模型预测评估失败: {str(e)}")

    # # 5. 测试: ROC曲线绘制
    # print("\n5. 测试: ROC曲线绘制...")
    # try:
    #     if 'model' in locals() and 'X_test' in locals() and 'y_test' in locals():
    #         plot_roc_curve(model, X_test, y_test, "Test Model", save_dir='../data/results')
    #         print("ROC曲线绘制完成")
    #         logger.info("ROC曲线绘制完成")
    #     else:
    #         print("缺少必要数据，跳过ROC曲线绘制测试")
    #         logger.warning("缺少必要数据，跳过ROC曲线绘制测试")
    # except Exception as e:
    #     print(f"ROC曲线绘制失败: {str(e)}")
    #     logger.error(f"ROC曲线绘制失败: {str(e)}")

    # # 6. 测试: 逐行预测功能
    # print("\n6. 测试: 逐行预测功能...")
    # try:
    #     # 检查是否存在test2.csv文件
    #     test2_path = '../data/test2.csv'
    #     if os.path.exists(test2_path):
    #         if 'model' in locals():
    #             print(f"   开始对 {test2_path} 进行逐行预测...")
    #             predictions_df = predict_row_by_row(model, test2_path, logger)
    #             if predictions_df is not None:
    #                 print("    逐行预测完成")
    #                 logger.info("逐行预测测试完成")
    #             else:
    #                 print("   逐行预测失败")
    #                 logger.error("逐行预测测试失败")
    #         else:
    #             print("    缺少模型，跳过逐行预测测试")
    #             logger.warning("缺少模型，跳过逐行预测测试")
    #     else:
    #         print(f"    文件 {test2_path} 不存在，跳过逐行预测测试")
    #         logger.warning(f"文件 {test2_path} 不存在，跳过逐行预测测试")
    # except Exception as e:
    #     print(f"    逐行预测测试失败: {str(e)}")
    #     logger.error(f"逐行预测测试失败: {str(e)}")

    # print("\n" + "="*60)
    # print("预测测试完成！")
    # print("="*60)
    # logger.info("预测测试完成！")
    main()