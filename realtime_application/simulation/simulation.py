import pandas as pd

class Temporary_Simulation:
    def __init__(self):
        self.simulation_df = pd.read_csv(r"./simulation/data/real_time_org_samsung_1/012.csv")

        # rename column from clinical dataset to vital DB dataset
        self.simulation_df.rename(
            columns={'ECG_II': 'SNUADC/ECG_II',
                     'NIBP_SBP': 'Solar8000/NIBP_SBP',
                     'NIBP_MBP': 'Solar8000/NIBP_MBP',
                     'NIBP_DBP': 'Solar8000/NIBP_DBP',
                     'PLETH': 'SNUADC/PLETH',
                     'MAC': 'Primus/MAC',
                     'CO2': 'Primus/CO2',
                     'BIS': 'BIS/BIS'}, inplace=True
        )

        #  'Time' 열을 삭제
        self.simulation_df = self.simulation_df.drop(columns=['Time'])

        # reorder column from clinical dataset to vital DB dataset
        self.simulation_df = self.simulation_df[
            ['SNUADC/ECG_II', 'Solar8000/NIBP_SBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP', 'SNUADC/PLETH', 'Primus/MAC', 'Primus/CO2', 'BIS/BIS']]

        first_idx = self.simulation_df['Solar8000/NIBP_MBP'].first_valid_index()
        last_idx = self.simulation_df['Solar8000/NIBP_MBP'].last_valid_index()
        self.simulation_df = self.simulation_df.loc[first_idx:last_idx]
        self.simulation_df.reset_index(drop=True, inplace=True)

        # 모든 데이터셋을 사용할 때까지 무한 반복
        self.continue_looping = True

    def extract_data(self):
        # 첫번째(1~100) 행 추출 후, 삭제
        first_row_list = self.simulation_df.iloc[0:100, :]
        self.simulation_df.drop(index=range(0, 100), inplace=True)
        self.simulation_df.reset_index(drop=True, inplace=True)

        if len(self.simulation_df) < 100:
            self.continue_looping = False

        disconnected_features = []

        return first_row_list, disconnected_features