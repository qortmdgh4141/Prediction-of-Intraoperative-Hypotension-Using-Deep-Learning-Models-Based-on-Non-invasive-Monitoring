import torch
import numpy as np
import pandas as pd

from ml_component.model.attention_galr import XGALR

class Inference:
    def __init__(self):
        multihead_GALR_model_settings = {
            'num_features': 8,  # MAC is dropped. 8 -> 7
            'embedding_linear_bias': False,
            'num_heads': 4,
            'batch_first': True,
            'num_classes': 2,

            'multihead_embedding_dim': 300,  # embedding on feature
            'multihead_sequences': 3000,
            'multihead_axis': 1,

            'galr_embedding_dim': 8,  # embedding on time 8->7
            'galr_axis': 0,

            'galr_chunk_size': 100,
            'galr_hop_size': 100,

            'galr_hidden_channels': 32,
            'galr_num_blocks': 3,
            'galr_bidirectional': False,
            'galr_eps': 1e-12,
            'galr_dropout': 1e-1,

            'temperature': 0.5,
            'save_attn': False
        }
        param = torch.load(r'./ml_component/preprocessed_file/seoul_national_university(best-model-9-52500).pt')
        state_dict = param['state_dict']

        self.model = XGALR(**multihead_GALR_model_settings)
        self.model.load_state_dict(state_dict)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device).type(torch.float64)
        self.model.eval()

    def predict(self, x):
        with torch.no_grad():
            input_x = x.to(self.device).type(torch.float64)
            predicted = self.model(input_x)
            predict_prob = torch.softmax(predicted, dim=1)
            low_prob = predict_prob[:, 1].cpu().numpy()  # 저혈압 발생확률 체크

            return low_prob

    def realtime_predict(self, realtime_dataset_ins):
        torch_input = realtime_dataset_ins.getitem()
        low_prob = self.predict(x=torch_input)

        return low_prob

    def init_data_wait_30s(self, realtime_dataset_ins, simulation_ins):
        first_null = True
        while True:
            # 맨 처음은 Null 값이 아닌 new_data 데이터셋 가져와야 함
            new_data, disconnected_features = simulation_ins.extract_data()
            if first_null:
                if np.isnan(new_data.iloc[0, :]).any():
                    continue
                else:
                    first_null = False

            # new_data 데이터셋 추가
            realtime_dataset_ins.input_df = pd.concat([realtime_dataset_ins.input_df, new_data], ignore_index=True)

            # null 값에 대해 보간법 적용
            if np.isnan(new_data.values).any():
                realtime_dataset_ins.input_df.fillna(method="ffill", inplace=True)

            # 30초 입력데이터가 dataframe에 쌓일 때까지 반복
            if len(realtime_dataset_ins.input_df) == realtime_dataset_ins.time_seq:
                print(f"DataFrame has reached {len(realtime_dataset_ins.input_df)} rows. Continue with the next steps.")
                break
            else:
                print(f"Waiting for {realtime_dataset_ins.time_seq - len(realtime_dataset_ins.input_df)} more rows to be added to the DataFrame...")