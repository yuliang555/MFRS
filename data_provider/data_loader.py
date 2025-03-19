import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.reference import ReferenceSeries
from utils.alignment import SynchronousAlignment
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', 
                 cycle=None, rs_type='sin', Q=0, manual_add=[], L=1000):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
            self.rs_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            self.rs_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.rs_type = rs_type
        self.Q = Q
        self.manual_add = manual_add
        self.L = L
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        
        reference_series = ReferenceSeries(data, self.rs_type, self.Q, pr=True if self.set_type == 0 else False)        
        reference_series.pbp_extractor()
        reference_series.base_pattern_extractor(self.manual_add)
        data_rs = reference_series.reference_series_generator()
        
        self.synchronous_alignment = SynchronousAlignment(self.seq_len, data[:self.L], data_rs[:self.L])
        
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_rs = data_rs[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])
        
        rs_x = self.data_rs[index:index + self.rs_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, rs_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv', target='OT', scale=True, timeenc=0, freq='t', 
                 cycle=None, rs_type='sin', Q=0, manual_add=[], L=1000):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
            self.rs_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            self.rs_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
                
        self.rs_type = rs_type
        self.Q = Q
        self.manual_add = manual_add        
        self.L = L      

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        
        reference_series = ReferenceSeries(data, self.rs_type, self.Q, pr=True if self.set_type == 0 else False)        
        reference_series.pbp_extractor()
        reference_series.base_pattern_extractor(self.manual_add)
        data_rs = reference_series.reference_series_generator()
        
        self.synchronous_alignment = SynchronousAlignment(self.seq_len, data[:self.L], data_rs[:self.L])
        
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_rs = data_rs[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])
        
        rs_x = self.data_rs[index:index + self.rs_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, rs_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', 
                 cycle=None, rs_type='sin', Q=0, manual_add=[], L=1000):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
            self.rs_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            self.rs_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        
        
        self.rs_type = rs_type
        self.Q = Q
        self.manual_add = manual_add         
        self.L = L       

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        
        reference_series = ReferenceSeries(data, self.rs_type, self.Q, pr=True if self.set_type == 0 else False)        
        reference_series.pbp_extractor()
        reference_series.base_pattern_extractor(self.manual_add)
        data_rs = reference_series.reference_series_generator()
        
        self.synchronous_alignment = SynchronousAlignment(self.seq_len, data[:self.L], data_rs[:self.L])


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_rs = data_rs[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])
        
        rs_x = self.data_rs[index:index + self.rs_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, rs_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', 
                 cycle=None, rs_type='sin', Q=0, manual_add=[], L=1000):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.rs_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        
        
        self.rs_type = rs_type
        self.Q = Q
        self.manual_add = manual_add         
        self.L = L       

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        
        reference_series = ReferenceSeries(data, self.rs_type, self.Q, pr=True if self.set_type == 0 else False)        
        reference_series.pbp_extractor()
        reference_series.base_pattern_extractor(self.manual_add)
        data_rs = reference_series.reference_series_generator()
        
        self.synchronous_alignment = SynchronousAlignment(self.seq_len, data[:self.L], data_rs[:self.L])


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_rs = data_rs[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])
        
        rs_x = self.data_rs[index:index + self.rs_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, rs_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Synthesis(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h',
                 cycle=None, rs_type='sin', Q=0, manual_add=[], L=1000):
        
        self.seq_len = size[0]
        self.refer_len = size[1]
        self.pred_len = size[2]
        self.rs_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
                
        self.rs_type = rs_type
        self.Q = Q
        self.manual_add = manual_add         
        self.L = L       

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_data = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        LL = len(df_data)

        num_train = int(LL * 0.7)       
        num_test = int(LL * 0.2)
        num_vali = LL - num_train - num_test
        border1s = [0, num_train - self.seq_len, LL - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, LL]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]        

        if self.scale:
            train_data = df_data[:num_train]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        
        self.data = data[border1:border2]            
        
        
        reference_series = ReferenceSeries(data, self.rs_type, self.Q, pr=True if self.set_type == 0 else False)        
        reference_series.pbp_extractor()
        reference_series.base_pattern_extractor(self.manual_add)
        data_rs = reference_series.reference_series_generator()
        
        self.synchronous_alignment = SynchronousAlignment(self.seq_len, data[:self.L], data_rs[:self.L])


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_rs = data_rs[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
       
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])
        
        rs_x = self.data_rs[index:index + self.rs_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, rs_x

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# # ## add cycle
# # class Dataset_Pred(Dataset):
# #     def __init__(self, root_path, flag='pred', size=None,
# #                  features='S', data_path='ETTh1.csv',
# #                  target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
# #         # size [seq_len, label_len, pred_len]
# #         # info
# #         if size == None:
# #             self.seq_len = 24 * 4 * 4
# #             self.label_len = 24 * 4
# #             self.pred_len = 24 * 4
# #         else:
# #             self.seq_len = size[0]
# #             self.label_len = size[1]
# #             self.pred_len = size[2]
# #         # init
# #         assert flag in ['pred']

# #         self.features = features
# #         self.target = target
# #         self.scale = scale
# #         self.inverse = inverse
# #         self.timeenc = timeenc
# #         self.freq = freq
# #         self.cols = cols
# #         self.root_path = root_path
# #         self.data_path = data_path
# #         self.__read_data__()

# #     def __read_data__(self):
# #         self.scaler = StandardScaler()
# #         df_raw = pd.read_csv(os.path.join(self.root_path,
# #                                           self.data_path))
# #         '''
# #         df_raw.columns: ['date', ...(other features), target feature]
# #         '''
# #         if self.cols:
# #             cols = self.cols.copy()
# #             cols.remove(self.target)
# #         else:
# #             cols = list(df_raw.columns)
# #             cols.remove(self.target)
# #             cols.remove('date')
# #         df_raw = df_raw[['date'] + cols + [self.target]]
# #         border1 = len(df_raw) - self.seq_len
# #         border2 = len(df_raw)

# #         if self.features == 'M' or self.features == 'MS':
# #             cols_data = df_raw.columns[1:]
# #             df_data = df_raw[cols_data]
# #         elif self.features == 'S':
# #             df_data = df_raw[[self.target]]

# #         if self.scale:
# #             self.scaler.fit(df_data.values)
# #             data = self.scaler.transform(df_data.values)
# #         else:
# #             data = df_data.values

# #         tmp_stamp = df_raw[['date']][border1:border2]
# #         tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
# #         pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

# #         df_stamp = pd.DataFrame(columns=['date'])
# #         df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
# #         if self.timeenc == 0:
# #             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
# #             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
# #             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
# #             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
# #             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
# #             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
# #             data_stamp = df_stamp.drop(['date'], 1).values
# #         elif self.timeenc == 1:
# #             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
# #             data_stamp = data_stamp.transpose(1, 0)

# #         self.data_x = data[border1:border2]
# #         if self.inverse:
# #             self.data_y = df_data.values[border1:border2]
# #         else:
# #             self.data_y = data[border1:border2]
# #         self.data_stamp = data_stamp

# #     def __getitem__(self, index):
# #         s_begin = index
# #         s_end = s_begin + self.seq_len
# #         r_begin = s_end - self.label_len
# #         r_end = r_begin + self.label_len + self.pred_len

# #         seq_x = self.data_x[s_begin:s_end]
# #         if self.inverse:
# #             seq_y = self.data_x[r_begin:r_begin + self.label_len]
# #         else:
# #             seq_y = self.data_y[r_begin:r_begin + self.label_len]
# #         seq_x_mark = self.data_stamp[s_begin:s_end]
# #         seq_y_mark = self.data_stamp[r_begin:r_end]

# #         return seq_x, seq_y, seq_x_mark, seq_y_mark

# #     def __len__(self):
# #         return len(self.data_x) - self.seq_len + 1

# #     def inverse_transform(self, data):
# #         return self.scaler.inverse_transform(data)


# class Dataset_PEMS(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', cycle=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cycle = cycle

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         data_file = os.path.join(self.root_path, self.data_path)
#         data = np.load(data_file, allow_pickle=True)
#         data = data['data'][:, :, 0]

#         num_train = int(len(data) * 0.6)
#         num_test = int(len(data) * 0.2)
#         num_valid = int(len(data) * 0.2)
#         border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_valid, len(data)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.scale:
#             train_data = data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data)
#             data = self.scaler.transform(data)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]

#         # add cycle
#         self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1))
#         seq_y_mark = torch.zeros((seq_x.shape[0], 1))

#         cycle_index = torch.tensor(self.cycle_index[s_end])
#         return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

