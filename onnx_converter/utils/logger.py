# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/10 15:38
# @File     : logger.py

# logging: record config, node, layer, graph, quantize, operators, correct
import logging

logger_initialized = {}

NOTSET, DEBUG, INFO = 0, 10, 20
WARNING, ERROR, FATAL = 30, 40, 50


class Logging(object):
    def __init__(self, **kwargs):
        self.NOTSET, self.DEBUG, self.INFO = 0, 10, 20
        self.WARNING, self.ERROR, self.FATAL = 30, 40, 50

        self.log_level = kwargs.get('log_level', 20) if 'log_level' in kwargs.keys() else self.NOTSET
        self.logger = None  # logging.getLogger(name)
        self.log_dict = {0: 'info', 10: 'debug', 20: 'info',
                         30: 'warning', 40: 'error', 50: 'fatal'}

    @staticmethod
    def get_streamhandler():
        stream_handler = logging.StreamHandler()
        return [stream_handler]
        
    # name is namespace, in order to redefine logger
    # log_file: log file name, if not exist, this function will build new one
    @staticmethod
    def get_root_logger(self, name, log_file='', file_mode='w', level=logging.INFO, stdout=True): # type: ignore
        self.log_level = level
        
        self.logger = logging.getLogger(name)
        if name in logger_initialized:
            return self.logger

        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return self.logger

        handlers = []
        if stdout:
            handlers.extend(Logging.get_streamhandler())

        if log_file is not None:
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(self.log_level)
            self.logger.addHandler(handler)

        self.logger.setLevel(self.log_level)

        logger_initialized[name] = True

        return self.logger

    # def logging(self, msg):
    #     eval('self.'+self.log_dict[self.log_level])(msg)

    def set_level(self, log_level=10):
        self.log_level = log_level
        self.logger.setLevel(log_level) # type: ignore

    def get_logger(self):
        return self.logger

    def info(self, msg):
        self.logger.info('\n'+msg) # type: ignore

    def error(self, msg):
        self.logger.error('\n'+msg) # type: ignore

    def fatal(self, msg):
        self.logger.critical('\n' + msg) # type: ignore

    def warning(self, msg):
        self.logger.warning('\n'+msg) # type: ignore

    def exception(self, msg):
        self.logger.exception('\n'+msg) # type: ignore

    def debug(self, msg):
        self.logger.debug('\n'+msg) # type: ignore

    def print_log(self, msg, logger=None, level=logging.INFO):
        if logger is None:
            print(msg)
        elif isinstance(logger, logging.Logger):
            logger.log(level, msg)
        elif logger == 'silent':
            pass
        elif isinstance(logger, str):
            _logger = self.get_root_logger(logger) # type: ignore
            _logger.log(level, msg)
        else:
            raise TypeError(
                'logger should be either a logging.Logger object, str, '
                f'"silent" or None, but got {type(logger)}')


if __name__ == '__main__':
    logger = Logging()
    logger.get_root_logger(name='test', log_file='/home/shiqing/Downloads/test.log') # type: ignore
    logger.info('debug for graph save pretty text!')
    logger.info('test')
