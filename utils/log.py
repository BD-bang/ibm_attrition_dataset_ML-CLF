# pylint: disable=line-too-long
# -*- coding: utf-8 -*-
import logging
import os

# 自定义的 日志工具类.
class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, root_path, log_name, level='info', fmt='%(asctime)s - %(levelname)s: %(message)s'):
        # 指定日志保存的路径
        self.root_path = root_path

        # 初始logger名称和格式
        self.log_name = log_name

        # 初始格式
        self.fmt = fmt

        # 先声明一个 Logger 对象
        self.logger = logging.getLogger(log_name)

        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))

    def get_logger(self):
        # 指定对应的 Handler 为 FileHandler 对象， 这个可适用于多线程情况
        path = os.path.join(self.root_path, 'log')
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, self.log_name + '.log')
        rotate_handler = logging.FileHandler(file_name, encoding="utf-8", mode="a")

        # Handler 对象 rotate_handler 的输出格式
        formatter = logging.Formatter(self.fmt)
        rotate_handler.setFormatter(formatter)

        # 将rotate_handler添加到Logger
        self.logger.addHandler(rotate_handler)

        return self.logger


# 测试的 Logger日志类的 添加数据到日志的使用方法.
if __name__ == '__main__':
    # 创建日志工具类.
    # 参1: 存储日志文件的(父目录), 一会后续会默认拼接上 log
    # 参2: 日志文件的名称, 一会后续会默认拼接上 .log
    logger = Logger('../', 'test01').get_logger()
    # 尝试往日志文件中写入数据.
    logger.info('这个是普通日志信息, 例如: **对象创建成功**')
    logger.error('这个是错误日志信息, 例如: **对象创建失败** 或者 No Such File Or Directory!...')
