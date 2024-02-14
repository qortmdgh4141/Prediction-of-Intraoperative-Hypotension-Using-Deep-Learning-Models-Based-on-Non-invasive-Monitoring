import torch
import pickle
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class Realtime_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.time_seq = 3000  # 30초 : 현재 시점 기준 이전 30초간의 입력데이터
        self.columns_list = ['SNUADC/ECG_II', 'Solar8000/NIBP_SBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP', 'SNUADC/PLETH', 'Primus/MAC', 'Primus/CO2', 'BIS/BIS']
        self.input_df = pd.DataFrame(columns=self.columns_list) # 30초 입력데이터 dataframe

        with open(r"./ml_component/preprocessed_file/newminmax_scaler.pkl", "rb") as file:
            self.scaler = pickle.load(file) # 8 feature

    def append_to_csv(self, new_data):
        # 데이터프레임의 마지막 행에 새로운 데이터 추가
        self.input_df = pd.concat([self.input_df, new_data], ignore_index=True)

        # 첫번째(1~100) 행 삭제
        self.input_df.drop(index=range(0, 100), inplace=True)
        self.input_df.reset_index(drop=True, inplace=True)

        # null 값에 대해 보간법 적용
        if np.isnan(new_data.values).any():
            self.input_df.fillna(method="ffill", inplace=True)

    def getitem(self):
        np_scaled_input = self.scaler.transform(self.input_df)
        np_scaled_input = np.delete(np_scaled_input, 5, axis=1)  # dropped 'Primus/MAC'
        torch_scaled_input = torch.tensor(np_scaled_input, dtype=torch.float64).unsqueeze(0)

        return torch_scaled_input