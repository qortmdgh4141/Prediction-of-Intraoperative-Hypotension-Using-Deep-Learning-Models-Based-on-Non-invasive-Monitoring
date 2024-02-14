# -*- coding: utf-8 -*-
################################################################################
## Form generated from reading UI file 'final_ui_mainIkrqFf.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
import time
import pyqtgraph as pg

from QLed import QLed
from PyQt5.QtGui import QFont, QPalette, QColor, QBrush
from PyQt5.QtCore import Qt, QSize, QMetaObject, QCoreApplication
from PyQt5.QtWidgets import QSizePolicy, QFrame, QWidget, QLayout, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLCDNumber, QSpacerItem, QProgressBar

class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        """ override 하여, tick 옆에 써지는 문자를 원하는대로 수정함.
            values --> x축 값들   ; 숫자로 이루어진 Itarable data --> ex) List[int]
        """
        return [time.strftime("%H:%M:%S", time.localtime(local_time)) for local_time in values]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.excel_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_data.xlsx"  # 현재 날짜 및 시간을 기반으로 한 파일명 설정
        # MainWindow : QMainWindow
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1800, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(1800, 600))
        MainWindow.setSizeIncrement(QSize(0, 0))
        MainWindow.setWindowOpacity(1.000000000000000)
        MainWindow.setLayoutDirection(Qt.LeftToRight)

        # central_widget : QWidget
        self.central_widget = QWidget(MainWindow)
        self.central_widget.setObjectName(u"central_widget")
        sizePolicy.setHeightForWidth(self.central_widget.sizePolicy().hasHeightForWidth())
        self.central_widget.setSizePolicy(sizePolicy)
        self.central_widget.setMinimumSize(QSize(0, 0))
        font = QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.central_widget.setFont(font)
        self.gridLayout_2 = QGridLayout(self.central_widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(13, -1, -1, -1)
        self.drop_shadow_widget = QWidget(self.central_widget)

        # drop_shadow_widget : QWidget
        self.drop_shadow_widget.setObjectName(u"drop_shadow_widget")
        self.drop_shadow_widget.setEnabled(False)
        sizePolicy.setHeightForWidth(self.drop_shadow_widget.sizePolicy().hasHeightForWidth())
        self.drop_shadow_widget.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setFamily(u"Bernard MT Condensed")
        font1.setPointSize(13)
        font1.setBold(True)
        font1.setItalic(True)
        font1.setWeight(75)
        font1.setKerning(False)
        self.drop_shadow_widget.setFont(font1)
        self.drop_shadow_widget.setContextMenuPolicy(Qt.NoContextMenu)
        self.drop_shadow_widget.setLayoutDirection(Qt.LeftToRight)
        self.drop_shadow_widget.setAutoFillBackground(False)
        self.drop_shadow_widget.setStyleSheet(u"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(42, 44, 111, 255), stop:0.521368 rgba(28, 29, 73, 255));\n"
"border-radius: 10px;")

        # verticalLayout : QVBoxLayout
        self.verticalLayout = QVBoxLayout(self.drop_shadow_widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(-1, 13, -1, -1)

        # parent_frame : QFrame
        self.parent_frame = QFrame(self.drop_shadow_widget)
        self.parent_frame.setObjectName(u"parent_frame")
        self.parent_frame.setFrameShape(QFrame.StyledPanel)
        self.parent_frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.parent_frame)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")

        # title_gridLayout : QGridLayout
        self.title_gridLayout = QGridLayout()
        self.title_gridLayout.setObjectName(u"title_gridLayout")
        self.title_label = QLabel(self.parent_frame)
        self.title_label.setObjectName(u"title_label")
        self.title_label.setEnabled(False)
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.title_label.sizePolicy().hasHeightForWidth())
        self.title_label.setSizePolicy(sizePolicy1)
        self.title_label.setMaximumSize(QSize(16777215, 50))
        self.title_label.setBaseSize(QSize(0, 3))
        font2 = QFont()
        font2.setFamily(u"Arial")
        font2.setPointSize(20)
        font2.setBold(True)
        font2.setItalic(False)
        font2.setUnderline(False)
        font2.setWeight(75)
        font2.setStrikeOut(False)
        font2.setKerning(True)
        font2.setStyleStrategy(QFont.PreferDefault)

        # title_label : QLabel
        self.title_label.setFont(font2)
        self.title_label.setFocusPolicy(Qt.ClickFocus)
        self.title_label.setContextMenuPolicy(Qt.NoContextMenu)
        self.title_label.setLayoutDirection(Qt.LeftToRight)
        self.title_label.setAutoFillBackground(False)
        self.title_label.setStyleSheet(u"color: rgb(60, 231, 195);")
        self.title_label.setInputMethodHints(Qt.ImhNone)
        self.title_label.setFrameShape(QFrame.NoFrame)
        self.title_label.setFrameShadow(QFrame.Plain)
        self.title_label.setLineWidth(0)
        self.title_label.setMidLineWidth(0)
        self.title_label.setTextFormat(Qt.PlainText)
        self.title_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.title_label.setWordWrap(False)
        self.title_label.setMargin(0)
        self.title_label.setIndent(0)
        self.title_label.setOpenExternalLinks(False)
        self.title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.title_gridLayout.addWidget(self.title_label, 0, 1, 1, 1)

        # rgb_btn_frame : QFrame
        self.rgb_btn_frame = QFrame(self.parent_frame)
        self.rgb_btn_frame.setObjectName(u"rgb_btn_frame")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.rgb_btn_frame.sizePolicy().hasHeightForWidth())
        self.rgb_btn_frame.setSizePolicy(sizePolicy2)
        self.rgb_btn_frame.setMaximumSize(QSize(16777215, 50))
        self.rgb_btn_frame.setLayoutDirection(Qt.RightToLeft)
        self.rgb_btn_frame.setFrameShape(QFrame.StyledPanel)
        self.rgb_btn_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.rgb_btn_frame)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(-1, -1, 10, -1)
        self.btn_green = QPushButton(self.rgb_btn_frame)
        self.btn_green.setObjectName(u"btn_green")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.btn_green.sizePolicy().hasHeightForWidth())

        # btn_green : QpushButton
        self.btn_green.setSizePolicy(sizePolicy3)
        self.btn_green.setMinimumSize(QSize(16, 16))
        self.btn_green.setMaximumSize(QSize(17, 17))
        self.btn_green.setStyleSheet(u"QPushButton {\n"
"	border: none;\n"
"	border-radius: 8px;	\n"
"	background-color: rgb(85, 255, 127);\n"
"}\n"
"QPushButton:hover {	\n"
"	background-color: rgba(85, 255, 127, 150);\n"
"}")

        self.horizontalLayout_3.addWidget(self.btn_green)

        # btn_orange : QpushButton
        self.btn_orange = QPushButton(self.rgb_btn_frame)
        self.btn_orange.setObjectName(u"btn_orange")
        sizePolicy3.setHeightForWidth(self.btn_orange.sizePolicy().hasHeightForWidth())
        self.btn_orange.setSizePolicy(sizePolicy3)
        self.btn_orange.setMinimumSize(QSize(16, 16))
        self.btn_orange.setMaximumSize(QSize(17, 17))
        self.btn_orange.setStyleSheet(u"QPushButton {\n"
"	border: none;\n"
"	border-radius: 8px;		\n"
"	background-color: rgb(255, 170, 0);\n"
"}\n"
"QPushButton:hover {	\n"
"	background-color: rgba(255, 170, 0, 150);\n"
"}")

        self.horizontalLayout_3.addWidget(self.btn_orange)

        # btn_red : QpushButton
        self.btn_red = QPushButton(self.rgb_btn_frame)
        sizePolicy3.setHeightForWidth(self.btn_red.sizePolicy().hasHeightForWidth())
        self.btn_red.setSizePolicy(sizePolicy3)
        self.btn_red.setMinimumSize(QSize(16, 16))
        self.btn_red.setMaximumSize(QSize(17, 17))
        self.btn_red.setStyleSheet(u"QPushButton {\n"
"	border: none;\n"
"	border-radius: 8px;		\n"
"	background-color: rgb(255, 0, 0);\n"
"}\n"
"QPushButton:hover {		\n"
"	background-color: rgba(255, 0, 0, 150);\n"
"}")
        self.horizontalLayout_3.addWidget(self.btn_red)

        self.title_gridLayout.addWidget(self.rgb_btn_frame, 0, 0, 1, 1)

        # title_line : Line
        self.title_line = QFrame(self.parent_frame)
        self.title_line.setObjectName(u"title_line")
        self.title_line.setFrameShape(QFrame.HLine)
        self.title_line.setFrameShadow(QFrame.Sunken)
        self.title_gridLayout.addWidget(self.title_line, 1, 0, 1, 2)


        self.verticalLayout_3.addLayout(self.title_gridLayout)

        # led_horizontalLayout : QVBoxLayout
        self.led_horizontalLayout = QHBoxLayout()
        self.led_horizontalLayout.setObjectName(u"led_horizontalLayout")
        min_led_size = (350, 350)
        max_led_size = (400, 400)

        spacer = QSpacerItem(75, 50, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.led_horizontalLayout.addItem(spacer)

        self.red_led = QLed(self, onColour=QLed.Red, shape=QLed.Circle)
        self.red_led.value = True
        self.red_led.setMinimumSize(min_led_size[0], min_led_size[1])
        self.red_led.setMaximumSize(max_led_size[0], max_led_size[1])
        self.led_horizontalLayout.addWidget(self.red_led)

        spacer = QSpacerItem(75, 50, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.led_horizontalLayout.addItem(spacer)

        self.orange_led = QLed(self, onColour=QLed.Orange, shape=QLed.Circle)
        self.orange_led.value = False
        self.orange_led.setMinimumSize(min_led_size[0], min_led_size[1])
        self.orange_led.setMaximumSize(max_led_size[0], max_led_size[1])
        self.led_horizontalLayout.addWidget(self.orange_led)

        spacer = QSpacerItem(75, 50, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.led_horizontalLayout.addItem(spacer)

        self.green_led = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.green_led.value = False
        self.green_led.setMinimumSize(min_led_size[0], min_led_size[1])
        self.green_led.setMaximumSize(max_led_size[0], max_led_size[1])
        self.led_horizontalLayout.addWidget(self.green_led)

        spacer = QSpacerItem(200, 50, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.led_horizontalLayout.addItem(spacer)

        # text_gridLayout : QGridLayout
        self.text_gridLayout = QGridLayout()
        self.text_gridLayout.setObjectName(u"text_gridLayout")
        font3 = QFont()
        font3.setFamily(u"Arial")
        font3.setPointSize(15)
        font3.setBold(True)
        font3.setItalic(False)
        font3.setUnderline(False)
        font3.setWeight(75)
        font3.setStrikeOut(False)
        font3.setKerning(True)
        font3.setStyleStrategy(QFont.PreferDefault)
        palette = QPalette()
        palette.setColor(QPalette.WindowText, QColor(120, 120, 120))  # 화이트(흰색)
        text_min_size = QSize(15, 10)
        text_max_size = QSize(180, 100)

        self.label_1 = QLabel(self.parent_frame)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_1.setFont(font3)
        self.label_1.setPalette(palette)
        self.label_1.setMinimumSize(text_min_size)
        self.label_1.setMaximumSize(text_max_size)
        self.text_gridLayout.addWidget(self.label_1, 1, 1, 1, 1)

        self.label_2 = QLabel(self.parent_frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_2.setFont(font3)
        self.label_2.setPalette(palette)
        self.label_2.setMinimumSize(text_min_size)
        self.label_2.setMaximumSize(text_max_size)
        self.text_gridLayout.addWidget(self.label_2, 2, 1, 1, 1)

        self.label_3 = QLabel(self.parent_frame)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_3.setFont(font3)
        self.label_3.setPalette(palette)
        self.label_3.setMinimumSize(text_min_size)
        self.label_3.setMaximumSize(text_max_size)
        self.text_gridLayout.addWidget(self.label_3, 3, 1, 1, 1)

        self.label_4 = QLabel(self.parent_frame)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_4.setFont(font3)
        self.label_4.setPalette(palette)
        self.label_4.setMinimumSize(text_min_size)
        self.label_4.setMaximumSize(text_max_size)
        self.text_gridLayout.addWidget(self.label_4, 4, 1, 1, 1)

        self.label_5 = QLabel(self.parent_frame)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_5.setFont(font3)
        self.label_5.setPalette(palette)
        self.label_5.setMinimumSize(text_min_size)
        self.label_5.setMaximumSize(text_max_size)
        self.text_gridLayout.addWidget(self.label_5, 5, 1, 1, 1)

        self.label_6 = QLabel(self.parent_frame)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_6.setFont(font3)
        self.label_6.setPalette(palette)
        self.label_6.setMinimumSize(text_min_size)
        self.label_6.setMaximumSize(text_max_size)
        self.text_gridLayout.addWidget(self.label_6, 6, 1, 1, 1)

        self.label_7 = QLabel(self.parent_frame)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_7.setFont(font3)
        self.label_7.setPalette(palette)
        self.label_7.setMinimumSize(text_min_size)
        self.label_7.setMaximumSize(text_max_size)
        self.text_gridLayout.addWidget(self.label_7, 7, 1, 1, 1)

        self.led_horizontalLayout.addLayout(self.text_gridLayout)

        self.verticalLayout_3.addLayout(self.led_horizontalLayout)

        # centerline_horizontalLayout : QVBoxLayout
        self.centerline_horizontalLayout = QVBoxLayout()
        self.centerline_horizontalLayout.setObjectName(u"centerline_horizontalLayout")

        # center_line : Line
        self.center_line = QFrame(self.parent_frame)
        self.center_line.setObjectName(u"center_line")
        font3 = QFont()
        font3.setFamily(u"Arial")
        font3.setPointSize(9)
        self.center_line.setFont(font3)
        self.center_line.setFrameShape(QFrame.HLine)
        self.center_line.setFrameShadow(QFrame.Sunken)
        self.centerline_horizontalLayout.addWidget(self.center_line)

        self.verticalLayout_3.addLayout(self.centerline_horizontalLayout)

        # probability_horizontalLayout : QHBoxLayout
        self.probability_horizontalLayout = QHBoxLayout()
        self.probability_horizontalLayout.setObjectName(u"probability_horizontalLayout")

        # 실시간 현재 시점 저혈압 확률 그래프 출력
        self.pw = pg.PlotWidget(
            labels={'left': 'Probability of Hypotension Occurrence (%)'},
            axisItems={'bottom': TimeAxisItem(orientation='bottom')}
        )
        background_color = QColor(0, 0, 0, 0)
        brush = QBrush(background_color)
        self.pw.setBackground(brush)
        self.pw.setTitle("Real-time Prediction of Hypotension Probability Over the Next 5 Minutes", **{'color': 'w', 'size': '10pt'})
        self.pw.setYRange(0, 100)
        time_data = int(time.time())
        self.pw.setXRange(time_data - 10, time_data + 1)
        self.pw.showGrid(x=True, y=True)
        self.pdi = self.pw.plot(pen=pg.mkPen(255, 0, 0, width=7), fillLevel=-0.9, brush=(255, 0, 0, 50))
        self.pw.setMaximumSize(1400, 350)
        self.plotData = {'x': [], 'y': []}
        self.probability_horizontalLayout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        self.probability_horizontalLayout.addWidget(self.pw)

        # 실시간 현재 시점 저혈압 확률 값 출력
        self.probability_lcdNumber = QLCDNumber(self.parent_frame)
        self.probability_lcdNumber.setMaximumSize(350, 350)
        self.probability_lcdNumber.setObjectName(u"probability_lcdNumber")
        self.probability_lcdNumber.setSmallDecimalPoint(False)
        self.probability_lcdNumber.setDigitCount(2)
        self.probability_lcdNumber.setMode(QLCDNumber.Dec)
        self.probability_lcdNumber.setSegmentStyle(QLCDNumber.Filled)
        self.probability_lcdNumber.setProperty("intValue", 80) # self.probability_lcdNumber.setProperty("value", 80.000000000000000)
        self.probability_horizontalLayout.addWidget(self.probability_lcdNumber)

        self.verticalLayout_3.addLayout(self.probability_horizontalLayout)

        self.verticalLayout.addWidget(self.parent_frame)

        self.gridLayout_2.addWidget(self.drop_shadow_widget, 5, 0, 1, 1)

        MainWindow.setCentralWidget(self.central_widget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWin", u"  Real-Time Hypotension Monitoring System", None))
        self.central_widget.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><br/></p></body></html>", None))
        self.title_label.setText(QCoreApplication.translate("MainWindow", u"  Real-Time Hypotension Monitoring System", None))
        self.btn_green.setToolTip("")
        self.btn_green.setText("")
        self.btn_orange.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><br/></p></body></html>", None))
        self.btn_orange.setText("")
        self.btn_red.setToolTip("")
        self.btn_red.setText("")
        self.label_1.setText(QCoreApplication.translate("MainWindow", u"  ⛔  ECG_II_WAV", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"  ⛔  CO2_WAV ", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"  ⛔  EEG_BIS", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"  ⛔  PLETH", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"  ⛔  NIBP_SYS", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"  ⛔  NIBP_MEAN", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"  ⛔  NIBP_DIA", None))