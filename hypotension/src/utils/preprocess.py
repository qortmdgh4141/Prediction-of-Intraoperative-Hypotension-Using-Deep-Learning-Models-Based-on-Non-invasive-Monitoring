from .. import *
from . import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from multiprocessing import Process

import glob
import h5py
import pickle

# change clinical column to vital db column
# dropped mac
def job(file: str, time_seq: int, time_delay: int, prediction_lag: int, dst_path: str, scaler) -> None:
    # Get CID
    cid = file.split("/")[-1].split(".")[0]
    # Read csv
    df = pd.read_csv(file)

    ##If dataset is clinical data:
    ##rename column from clinical dataset to vital DB dataset
    df.rename(columns={'ECG_II': 'SNUADC/ECG_II', 'NIBP_SBP': 'Solar8000/NIBP_SBP', 'NIBP_MBP':'Solar8000/NIBP_MBP',
                       'NIBP_DBP':'Solar8000/NIBP_DBP','PLETH':'SNUADC/PLETH', 'MAC':'Primus/MAC','CO2':'Primus/CO2',
                       'BIS':'BIS/BIS'}, inplace=True)

    ##reorder column from clinical dataset to vital DB dataset
    df = df[['Time','SNUADC/ECG_II', 'Solar8000/NIBP_SBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP',
                             'SNUADC/PLETH', 'Primus/MAC', 'Primus/CO2', 'BIS/BIS']]
    # Eliminate NAs
    df = eliminate_na(data_frame=df)
    # Labeling
    df = labeling(input_df=df)

    df.drop(columns=['Primus/MAC'])

    np_scaled = scaler.transform(df[df.columns[1:-1]])

    np_scaled = pd.DataFrame(np_scaled, columns=df.columns[1:-1], index=list(df.index.values))

    scaled_df = pd.concat((df.loc[:, ['Time']], np_scaled), axis=1)
    df = pd.concat((scaled_df, df.iloc[:, -1]), axis=1)

    # Data split
    data_split(cid, df, time_seq=time_seq, time_delay=time_delay, target_seq=prediction_lag, dst_path=dst_path)


def make_dataset(data_path: str, time_seq: int, time_step: int, target_seq: int, dst_path: str) -> None:
    """
    Making dataset from each original case data. Eliminating NA, Fill NA, and labeling are included.
    :param data_path: (str) Original case data directory path
    :param time_seq: (int) Input sequence length. (e.g., 3 minutes to observe for the prediction)
    :param time_step: (int) Steps for time observing sequences.
    :param target_seq: (int) A number of sequences for output target. (e.g., 5 minutes after input)
    :param dst_path: (str) Path for saving result.
    :return: None
    """
    file_list = glob.glob(os.path.join(data_path, "original/*.csv"))
    ##remove BIS outliers
    outliers = []
    with open("./outliers.txt") as f:
        for line in f:
            outliers.append(line.strip())

    file_list = [file for file in file_list if file not in outliers]
    ##outlier just removed from file list

    processes = []
    pbar = tqdm(file_list, desc='Data preprocessing')

    ##

    #scaler = normalization(file_list)

    # NOTE: Load from file.
    with open("./scaler/newminmax_scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    while file_list:
        file = file_list.pop()
        pbar.update(1)
        p = Process(target=job, args=(file, time_seq, time_step, target_seq, dst_path, scaler,))
        p.start()
        processes.append(p)

        if len(processes) >= os.cpu_count():
            while processes:
                _p = processes.pop()
                _p.join()

    # If process still left, flush...
    if processes:
        while processes:
            _p = processes.pop()
            _p.join()


def data_split(
        cid: str,
        dataframe: DataFrame,
        time_seq: int, time_delay: int, target_seq: int,
        dst_path: str) -> None:
    """
    Data split for train and test.
    Args:
        cid: (str) Case ID
        dataframe: (DataFrame) Pandas DataFrame to process.
        time_seq: (int) Observing sequences.
        time_delay: (int) Time lag.
        target_seq: (int) Prediction target.
        dst_path: (str) Destination path to save file.

    Returns: None

    """
    total_seq = time_seq + target_seq + time_delay
    data_dict = {
        'x': [],
        'y': []
    }

    # NOTE: Make X values and Y values.
    #pd_x = dataframe.iloc[:, 1:-1]

    # #use loc for unsorted clinical indexer
    pd_x = dataframe.loc[:, ['SNUADC/ECG_II', 'Solar8000/NIBP_SBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP',
                             'SNUADC/PLETH',  'Primus/CO2', 'BIS/BIS']]#dropped 'Primus/MAC',


    pd_y = dataframe.iloc[:, -1]
    pd_y = pd_y.replace({'normal': 0, 'low': 1, pd.NA: -1})

    # INFO: To numpy
    np_x = pd_x.to_numpy(dtype=np.float16)
    np_y = pd_y.to_numpy(dtype=np.int64)

    for i in range(0, len(np_x) - total_seq, time_delay):
        _x = np_x[i:i + time_seq]
        _y = np_y[i + time_seq + target_seq]

        ############################################################################
        # INFO: Exception mechanism. You may add HERE if another exception needed.
        if np.isnan(_x).any() or _y <= -1:
            continue
        ############################################################################

        data_dict['x'].append(_x)
        data_dict['y'].append(_y)

    # Save numpy using HDF5 format
    date_time = datetime.now().strftime("%Y-%m-%d")

    save_path = os.path.join(dst_path, date_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(os.path.join(save_path, "Datameta.txt")):
        with open(os.path.join(save_path, "Datameta.txt"), "w") as file:
            file.write("time_seq: {}\n".format(time_seq))
            file.write("target_seq: {}\n".format(target_seq))
            file.write("time_delay: {}\n".format(time_delay))
            file.write("scaler: {}\n".format("min_max"))

    save_path = os.path.join(save_path, "{}.hdf5".format(cid))

    with h5py.File(save_path, "w") as f:
        try:
            f.create_dataset('x', data=data_dict['x'])
            f.create_dataset('y', data=data_dict['y'])
        except Exception as e:
            print(e)
        finally:
            f.close()

# change clinical column to vital db column
# dropped mac
def normalization(file_list: list):
    """
    Normalize the data.
    :param file_list (str) File path
    :return: Dataframe, scaler
    """
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    if not os.path.exists('./scaler'):
        os.makedirs('./scaler')
    attributes = ["Time",
                  "SNUADC/ECG_II",
                  "Solar8000/NIBP_SBP",
                  "Solar8000/NIBP_MBP",
                  "Solar8000/NIBP_DBP",
                  "SNUADC/PLETH",
                  "Primus/MAC",
                  "Primus/CO2",
                  "BIS/BIS"]

    for file in tqdm(file_list, desc="Normalization"):
        #df = pd.read_csv(file, usecols=attributes)
        df = pd.read_csv(file)

        ##rename column from clinical dataset to vital DB dataset
        df.rename(
            columns={'ECG_II': 'SNUADC/ECG_II', 'NIBP_SBP': 'Solar8000/NIBP_SBP', 'NIBP_MBP': 'Solar8000/NIBP_MBP',
                     'NIBP_DBP': 'Solar8000/NIBP_DBP', 'PLETH': 'SNUADC/PLETH', 'MAC': 'Primus/MAC',
                     'CO2': 'Primus/CO2',
                     'BIS': 'BIS/BIS'}, inplace=True)
        ##reorder column from clinical dataset to vital DB dataset
        df = df[['Time', 'SNUADC/ECG_II', 'Solar8000/NIBP_SBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP',
                 'SNUADC/PLETH', 'Primus/MAC', 'Primus/CO2', 'BIS/BIS']]

        #drop MAC column (inexact column)
        df.drop(columns=['Primus/MAC'])

        # NOTE: except Time
        _ = scaler.partial_fit(df[df.columns[1:]])

    with open("./scaler/newminmax_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    return scaler


def eliminate_na(data_frame: DataFrame) -> DataFrame:
    """
    Eliminate NA on NIBP_SBP and MAC
    :param data_frame: (Dataframe) Input dataframe.
    :return: Dataframe
    """
    _data_frame = data_frame.copy(deep=True)

    # NOTE: 1. Extract valid range which Solar8000/NIBP_SBP value exist.
    first_idx = _data_frame['Solar8000/NIBP_SBP'].first_valid_index()
    last_idx = _data_frame['Solar8000/NIBP_SBP'].last_valid_index()
    #first_idx = _data_frame['NIBP_SBP'].first_valid_index()
    #last_idx = _data_frame['NIBP_SBP'].last_valid_index()

    _data_frame = _data_frame.loc[first_idx:last_idx]

    # NOTE: 2. Fill the value if NA exists. But not redundantly 300 times.
    _data_frame.fillna(method="ffill", inplace=True)
    _data_frame.fillna(method="bfill", inplace=True)

    # NOTE: Sort by Timeline
    _data_frame.sort_values(by=['Time'], inplace=True)

    return _data_frame


def labeling(input_df=None, dataset_path=None) -> DataFrame:
    """
    Labeling the target value
    :param input_df: (Dataframe) Input dataframe
    :param dataset_path: (str) Data path to save
    :return: Dataframe
    """

    if input_df is not None:
        df = input_df.copy(deep=True)
    elif dataset_path is not None:
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError("[labeling] Empty parameter.")

    # values = ["low", "high", "normal"]
    values = ["low", "normal", pd.NA]
    conditions = [
        (df['Solar8000/NIBP_SBP'] < 90),
        # (df['Solar8000/NIBP_SBP'] > 180),
        (df['Solar8000/NIBP_SBP'] >= 90),
        (df['Solar8000/NIBP_SBP'].isna())
    ]
    df['Target'] = np.select(conditions, values)

    return df
