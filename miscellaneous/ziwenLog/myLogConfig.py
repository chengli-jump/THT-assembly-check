import logging
from logging.handlers import RotatingFileHandler
import os
import time


class ConfigMyLog():
    """Wrap logging setting into a class

            Note that Loggers should NEVER be instantiated directly, but always through the module-level function 
            logging.getLogger(name). Multiple calls to getLogger() with the same name will always return a reference
            to the same Logger object.

            I just wrap it into a class for convenience.
    """

    def __init__(
            self, loggerName, logFileName='myProgram.log', logFolderPath=os.path.join(os.path.dirname(__file__), 'Logs'),
            withFolder=True, rotateLogs=True, maxBytes=10*1024*1024, backupCount=50, consoleLevel=logging.DEBUG, logLevel=logging.DEBUG):
        """Set up a logger. 

       Args:
            loggerName(str):the name of the getLogger(name)
            logFileName (str, optional): log file name. Defaults to 'default'.
            logFolderPath (str,optional): the path of the folder where put all the logs.Defaults to './Logs'
            withFolder (bool, optional): generate folder to contain all logs. Defaults to True.
            rotateLogs(bool,optional): rotate logs. avoid the size of the logs are too large.Defaults to True
            consoleLevel ([type], optional): Defaults to logging.DEBUG.
            logLevel ([type], optional): Defaults to logging.DEBUG.
        """
        self._loggerName = loggerName
        self.logger = logging.getLogger(self._loggerName)
        if withFolder:
            self._logFolderPath = os.path.join(logFolderPath, str(time.strftime("%Y%m%d%H%M%S", time.localtime())))
        else:
            self._logFolderPath = logFolderPath
        if not os.path.exists(self._logFolderPath):
            os.makedirs(self._logFolderPath)
        self._logFile = os.path.join(self._logFolderPath, logFileName)
        print(self._logFile)
        if rotateLogs:
            self.fileHandler = self.__setupRotateLogs(maxBytes, backupCount)
        else:
            self.fileHandler = logging.FileHandler(self._logFile, mode='a', encoding='utf8', delay=False)
        self.__setupDefaultLevel(consoleLevel, logLevel)

    def __setupDefaultLevel(self, consoleLevel, logLevel):
        # setup console.
        console = logging.StreamHandler()
        console.setLevel(consoleLevel)
        formatterConsole = logging.Formatter('%(name)-20s: %(levelname)-8s %(message)s')
        console.setFormatter(formatterConsole)
        logging.basicConfig(level=logLevel,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            handlers=[self.fileHandler, console])

    def __setupRotateLogs(self, maxBytes, backupCount):
        fh = RotatingFileHandler(self._logFile, maxBytes=maxBytes, backupCount=backupCount)
        return fh

    def give_me_a_logger(self):
        return self.logger


if __name__ == '__main__':
    # simply for test purpose.
    my_logger = ConfigMyLog('logger_main', logFileName='testMyLog.log',
                            maxBytes=3*1024, backupCount=3).give_me_a_logger()
    for i in range(40):
        my_logger.debug(str(i))
        my_logger.debug(r'It is only for test purpose')
