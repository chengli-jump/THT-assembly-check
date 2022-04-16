import sys
import os
from PyQt5 import QtWidgets
from controller.controller import MainWindow
from miscellaneous.ziwenLog.myLogConfig import ConfigMyLog

logFolderPath = os.path.join(os.path.dirname(__file__), 'logs')
my_logger = ConfigMyLog('logger_main', logFileName='THT_Check.log', logFolderPath=logFolderPath, withFolder=False,
                        maxBytes=30*1024, backupCount=3).give_me_a_logger()
my_logger.info('THT assembly component checking program started...')
app = QtWidgets.QApplication(sys.argv)
application = MainWindow()
application.show()
sys.exit(app.exec())
