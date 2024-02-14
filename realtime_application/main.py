import sys
import time
import random
import threading
import pandas as pd
import pyqtgraph as pg

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication

from gui.ui_mainwindow import Ui_MainWindow
from ml_component.inference import Inference
from ml_component.realtime_dataset import Realtime_Dataset
from simulation.vitaldb_streaming import VitalDB_Streaming
from simulation.simulation import Temporary_Simulation

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # 엑셀 파일에 저장할 데이터프레임 컬럼명 설정
        self.save_df = pd.DataFrame(columns=["Time", "Current Actual Status", "Model Output", "Predicted Status in 5 Minutes", "Prediction Matched", "Disconnected Features"])
        self.excel_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_data.xlsx"  # 현재 날짜 및 시간을 기반으로 한 파일명 설정
        self.save_flag = 0

    def uniform_model_run(self):
        low_prob = random.uniform(0, 100)
        future_state = "red" if low_prob > 53 else "orange" if low_prob > 41 else "green"
        return future_state, low_prob

    def simulation_model_run(self, simulation_ins, realtime_dataset_ins, inference_ins):
        new_data, disconnected_features = simulation_ins.extract_data()
        label = "low" if new_data['Solar8000/NIBP_SBP'].iloc[-1] < 90 else "normal"

        realtime_dataset_ins.append_to_csv(new_data=new_data)
        low_prob = inference_ins.realtime_predict(realtime_dataset_ins)
        low_prob = int(float(low_prob[0]) * 100)

        if low_prob > 53:
            future_state_tp = ("red", "low")
        elif low_prob > 41:
            future_state_tp = ("orange", "low")
        else:
            future_state_tp = ("green", "normal")

        return future_state_tp, low_prob, disconnected_features, label

    def add_data_to_dataframe(self, new_time_data, future_state, low_prob, disconnected_features, label):
        formatted_time = time.strftime('%H:%M:%S', time.localtime(new_time_data))
        # 실제 column명 : "Time", "Current Actual Status", "Model Output", "Predicted Status in 5 Minutes", "Prediction Matched", "Disconnected Features"
        new_row = {
            "Time": formatted_time, "Current Actual Status": label, "Model Output": low_prob,
            "Predicted Status in 5 Minutes": future_state, "Disconnected Features": disconnected_features,
        }
        self.save_df = pd.concat([self.save_df, pd.DataFrame([new_row])], ignore_index=True) # 새로운 행 추가

    def graph_get_data(self, simulation_ins, realtime_dataset_ins, inference_ins):
        # future_state, low_prob = self.uniform_model_run()
        future_state_tp, low_prob, disconnected_features, label = self.simulation_model_run(simulation_ins, realtime_dataset_ins, inference_ins)
        future_state_color, future_state = future_state_tp

        new_time_data = int(time.time())
        self.add_data_to_dataframe(new_time_data, future_state, low_prob, disconnected_features, label)

        total_rows = len(self.save_df)
        if self.save_flag == 60:
            if (total_rows - 300) > 0:
                for idx in range(total_rows - 300):  # 마지막 300행은 제외
                    self.save_df.loc[idx, 'Prediction Matched'] = self.save_df.loc[idx, 'Predicted Status in 5 Minutes'] == self.save_df.loc[idx + 300, 'Current Actual Status']

                # 마지막 300행의 'Prediction Matched' 값을 NaN으로 설정 (옵션)
                self.save_df.loc[total_rows - 300:, 'Prediction Matched'] = None
                self.save_df.to_excel(self.excel_filename, index=False, engine='openpyxl')
                self.save_flag = 0
                print(f"Save - {new_time_data}")
        else:
            self.save_flag += 1

        if future_state_color == "red":
            led_value = (True,False,False)
            symbol_color = (255, 0, 0, 50)
        elif future_state_color == "orange":
            led_value = (False, True, False)
            symbol_color = (255, 165, 0, 50)
        elif future_state_color == "green":
            led_value = (False, False, True)
            symbol_color = (0, 255, 0, 50)

        self.convert_led(led_value, symbol_color)
        self.update_labels_with_features(disconnected_features)  # 연결되지 않은 feature 경고 텍스트 출력
        self.update_plot(new_time_data, low_prob, symbol_color=symbol_color)
        self.probability_lcdNumber.setProperty("value", low_prob)

    def convert_led(self, led_value, symbol_color):
        self.red_led.value, self.orange_led.value, self.green_led.value = led_value
        self.pdi = self.pw.plot(pen=pg.mkPen(symbol_color[:3], width=7), fillLevel=-0.9, brush=symbol_color)

    def update_labels_with_features(self, disconnected_features):
        all_features = ['ECG_II_WAV', 'CO2_WAV', 'EEG_BIS', 'PLETH', 'NIBP_SYS', 'NIBP_MEAN', 'NIBP_DIA'] # 둘이 순서 일치해야함
        all_labels = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6, self.label_7] # 둘이 순서 일치해야함

        for i, feature_name in enumerate(all_features):
            indicator = "✅" if feature_name not in disconnected_features else "⛔"
            all_labels[i].setText(f"  {indicator}  {feature_name}")

    def update_plot(self, x, y, symbol_color):
        # 맨 처음 값을 출력할떄, 0으로 초기화
        if len(self.plotData['x']) == 0:
            for i in range(10):
                self.plotData['x'].append(x-10+i)
                self.plotData['y'].append(0)

        self.plotData['x'].append(x)
        self.plotData['y'].append(y)
        self.pw.setXRange(x - 10, x + 1)

        self.pdi.setData(self.plotData['x'][:-2], self.plotData['y'][:-2], symbol='o', symbolpen='o', symbolSize=18, symbolBrush=symbol_color)
        self.pdi.setData(self.plotData['x'][-2:], self.plotData['y'][-2:], symbol='o', symbolpen='o', symbolSize=18, symbolBrush=symbol_color)

if __name__ == "__main__" :
    temporary_simulation = True # 실제 수술장에서 필립스 장비와 연결하여 실행할때는, 반드시 False로 설정해야함!
    node_directory_path = r"C:\Users\qortm\PycharmProjects\realtime_application"
    node_script_path = "minimum_vitalserver.js"

    if temporary_simulation:
        simulation_ins = Temporary_Simulation()
    else:
        vitaldb_ins = VitalDB_Streaming(node_script_path, node_directory_path)
        node_thread = threading.Thread(target=vitaldb_ins.run_node_script)
        node_thread.start() # node thread 시작
        vitaldb_ins.start_web_driver() # chrome 웹페이지 연동'''

    realtime_dataset_ins = Realtime_Dataset()
    inference_ins = Inference()

    # 30초 입력데이터가 dataframe에 쌓일 때까지 반복
    if temporary_simulation:
        inference_ins.init_data_wait_30s(realtime_dataset_ins=realtime_dataset_ins, simulation_ins=simulation_ins)
    else:
        inference_ins.init_data_wait_30s(realtime_dataset_ins=realtime_dataset_ins, simulation_ins=vitaldb_ins)

    app = QApplication(sys.argv)
    app.setStyleSheet('QMainWindow{background-color: balck;border: 1px solid black;}')

    win = MainWindow()

    mytimer = QTimer()
    mytimer.start(1000)  # 1초마다 갱신 위함...
    if temporary_simulation:
        mytimer.timeout.connect(lambda: win.graph_get_data(simulation_ins=simulation_ins, realtime_dataset_ins=realtime_dataset_ins, inference_ins=inference_ins))
    else:
        mytimer.timeout.connect(lambda: win.graph_get_data(simulation_ins=vitaldb_ins, realtime_dataset_ins=realtime_dataset_ins,
                                   inference_ins=inference_ins))

    win.show()
    app.exec_()