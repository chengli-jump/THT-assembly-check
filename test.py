# -*- coding:utf-8 -*-
import logging
import datetime
from os import device_encoding
from sys import dllhandle

from numpy.lib.index_tricks import diag_indices_from


# filename：设置日志输出文件，以天为单位输出到不同的日志文件，以免单个日志文件日志信息过多，
# 日志文件如果不存在则会自动创建，但前面的路径如log文件夹必须存在，否则会报错
log_file = 'log.txt'
# level：设置日志输出的最低级别，即低于此级别的日志都不会输出
# 在平时开发测试的时候可以设置成logging.debug以便定位问题，但正式上线后建议设置为logging.WARNING，既可以降低系统I/O的负荷，也可以避免输出过多的无用日志信息
log_level = logging.WARNING
# format：设置日志的字符串输出格式
log_format = '%(asctime)s[%(levelname)s]: %(message)s'
logging.basicConfig(filename=log_file, level=log_level, format=log_format)
logger = logging.getLogger()

# 以下日志输出由于level被设置为了logging.WARNING，所以debug和info的日志不会被输出
logger.debug('This is a debug message!')
logger.info('This is a info message!'),dllhandledlvv;vz;;diag_indices_fromfllfd
device_encoding
d,dgbp.xlg
logger.warning('This is a warning message!')
logger.error('This is a error message!')
logger.critical('This is a critical message!')

