import subprocess
import numpy as np
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By

class VitalDB_Streaming:
    def __init__(self, node_script_name, node_directory_path):
        self.node_script_name = node_script_name
        self.node_directory_path = node_directory_path

    def run_node_script(self):
        subprocess.run(['node', self.node_script_name], cwd=self.node_directory_path)

    def start_web_driver(self):
        self.driver =webdriver.Chrome()

        # 브라우저 창 위치 및 크기 설정
        monitor_height = self.driver.execute_script("return window.screen.height")
        self.driver.set_window_size(0, monitor_height)
        self.driver.set_window_position(-10, 0)
        # 웹 페이지 열기
        self.driver.get("http://localhost:3000/")

    def stop_web_driver(self):
        if self.driver:
            self.driver.quit()

    def extract_data(self):
        # <div id="display"> 안의 내용 가져오기
        div_display = self.driver.find_element(By.ID, "display")
        content = div_display.text

        # 반드시 medical device (필립스)에서 설정한 이름과 동일해야함
        feature_dic = {key: [np.nan] * 100 for key in ['ECG_II_WAV', 'NIBP_SYS', 'NIBP_MEAN', 'NIBP_DIA', 'PLETH', 'CO2_WAV', 'EEG_BIS']}
        feature_dic = self.extract_values(content, feature_dic)

        df = pd.DataFrame(feature_dic)

        # 모두 np.nan으로 구성된 열의 열(column)명 추출
        disconnected_features= list(df.columns[df.isna().all()])

        df['MAC'] = 0

        # rename column from clinical dataset to vital DB dataset
        # 반드시 medical device (필립스)에서 설정한 이름과 동일해야함
        df.rename(
            columns={'ECG_II_WAV': 'SNUADC/ECG_II',
                     'NIBP_SYS': 'Solar8000/NIBP_SBP',
                     'NIBP_MEAN': 'Solar8000/NIBP_MBP',
                     'NIBP_DIA': 'Solar8000/NIBP_DBP',
                     'PLETH': 'SNUADC/PLETH',
                     'MAC': 'Primus/MAC',
                     'CO2_WAV': 'Primus/CO2',
                     'EEG_BIS': 'BIS/BIS'}, inplace=True
        )
        # reorder column from clinical dataset to vital DB dataset
        df = df[['SNUADC/ECG_II', 'Solar8000/NIBP_SBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP', 'SNUADC/PLETH', 'Primus/MAC', 'Primus/CO2', 'BIS/BIS']]

        return df, disconnected_features

    def extract_values(self, content, feature_dic):
        lines = content.split('\n')
        for line in lines:
            parts = line.split('=')
            feature_name = parts[0].strip()
            if feature_name in feature_dic:
                feature_values = parts[1].strip()

                # medical device에 connect에 문제 있을때
                if feature_values == '':
                    continue

                values = feature_values.split(',')
                values = [np.float32(value) for value in values]

                # medical device에 connect에 문제 있을때
                if len(values) != 1 and self.all_elements_equal(values):
                    continue

                # 100hz로 스케일링
                if len(values) > 100:
                    values = values[-100:]
                elif len(values) < 100:
                    values.extend([values[-1]] * (100 - len(values)))
                feature_dic[feature_name] = values

            else:
                continue

        return feature_dic

    def all_elements_equal(self, lst):
        first_element = lst[0]
        for element in lst:
            if element != first_element:
                return False
        # 모든 원소가 같으면 True를 반환
        return True

if __name__ == "__main__" :
    import time
    import threading
    node_directory_path = r"C:\Users\qortm\PycharmProjects\realtime_application"
    node_script_path = "minimum_vitalserver.js"

    # simulation_ins = Temporary_Simulation()
    vitaldb_ins = VitalDB_Streaming(node_script_path, node_directory_path)
    node_thread = threading.Thread(target=vitaldb_ins.run_node_script)
    node_thread.start() # node thread 시작
    vitaldb_ins.start_web_driver() # chrome 웹페이지 연동

    while True:
        time.sleep(1)
        df, disconnected_features = vitaldb_ins.extract_data()