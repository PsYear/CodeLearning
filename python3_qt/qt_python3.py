from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 设置窗口标题
        self.setWindowTitle('My First App')

        # 设置标签
        label = QLabel('Welcome to Shiyanlou!')
        # 设置标签显示在中央
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)
# 创建应用实例，通过 sys.argv 传入命令行参数
app = QApplication(sys.argv)
# 创建窗口实例
window = MainWindow()
# 显示窗口
window.show()
# 执行应用，进入事件循环
app.exec_()