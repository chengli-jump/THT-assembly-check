# myLogConfig
Configured logging for python script.

Loggers are never instantiated directly, but always through the module-level function logging.getLogger(name). Multiple calls to getLogger() with the same name will always return a reference to the same Logger object.

Even though, I prefer to wrap the config into a class.


#how to use:

from ziwenLog.myLogConfig import ConfigMyLog

my_logger = ConfigMyLog('logger_main', logFileName='testMyLog.log',maxBytes=3*1024, backupCount=3).give_me_a_logger()

for i in range(40):
    my_logger.debug(str(i))
    my_logger.debug(r'It is only for test purpose')
