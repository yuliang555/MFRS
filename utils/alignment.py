import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch




class SynchronousAlignment:
    def __init__(self, seq_len, train_data, rs_data, train_date=None):
        self.seq_len = seq_len
        self.L = train_data.shape[0] - seq_len
        self.N = min(train_data.shape[1], 30)
        
        self.train_data = train_data[:, :self.N] 
        self.rs_data = rs_data           
        self.train_date = train_date
        
        self.a = np.stack([self.train_data[step:step + self.seq_len] for step in range(self.L)], axis=0) # L * S * N
        self.rs = np.stack([self.rs_data[step:step + self.seq_len] for step in range(self.L)], axis=0) # L * S * N        


    def alignment(self, predict_data, device):
        self.B = predict_data.shape[0]                           
        self.predict_data = predict_data[:, :, :self.N].clone() 
        
        a = torch.tensor(self.a).unsqueeze(0).repeat(self.B, 1, 1, 1).to(device) # B * L * S * N         
        b = self.predict_data.unsqueeze(1).repeat(1, self.L, 1, 1).to(device) # B * L * S * N
        rs = torch.tensor(self.rs).float().to(device) # L * S * N
        
        mean_a = torch.mean(a, dim=2, keepdims=True) # B * L * S * N
        mean_b = torch.mean(b, dim=2, keepdims=True) # B * L * S * N
        cov = torch.sum((a - mean_a) * (b - mean_b), dim=2) / (self.seq_len - 1) # B * L * N
        std_a = torch.sqrt(torch.var(a, dim=2, unbiased=False) + 1e-5) # B * L * N
        std_b = torch.sqrt(torch.var(b, dim=2, unbiased=False) + 1e-5) # B * L * N

        self.step_corr = cov / (std_a * std_b) # B * L * N
        self.step = torch.argmax(torch.sum(self.step_corr, dim=-1), axis=-1) # B
        
        return rs[self.step]


    
    def verify_date(self, predict_date=None):
        assert self.train_date is not None and predict_date is not None, "Having no date information"
        
        
        for b in range(self.B):
            truth_date_obj = datetime.strptime(predict_date[b], "%Y-%m-%d %H:%M:%S")
            align_date_obj = datetime.strptime(self.train_date[self.step[b]], "%Y-%m-%d %H:%M:%S")
            
            truth_date = truth_date_obj.date()
            align_date = align_date_obj.date()
            
            truth_time = truth_date_obj.time()
            align_time = align_date_obj.time()
                    
            truth_week = truth_date_obj.weekday()    
            align_week = align_date_obj.weekday()
            
            print("\n***********************************************************\n")
            print(f"Truth:    date={truth_date}    time={truth_time}    week={truth_week}")
            print(f"Align:    date={align_date}    time={align_time}    week={align_week}")
            


    def draw_sliding(self, batch, channel):
        step_corr = self.step_corr.detach().cpu().numpy()
        predict_data = self.predict_data.detach().cpu().numpy()
        
        
        fig, ax = plt.subplots(3, 1, figsize=(6, 12))
        
        ax[0].plot(step_corr[batch, :, channel])
        ax[0].set_xlim(0, self.L)
        
        x = [i for i in range(self.step[batch], self.step[batch] + self.seq_len)]
        ax[1].plot(x, predict_data[batch, :, channel])
        ax[1].set_xlim(0, self.L)
        
        ax[2].plot(self.train_data[:, channel])
        ax[2].set_xlim(0, self.L)
                
        fig.savefig(f"{savepath}_{channel}.png")
        
        


if __name__ == "__main__":
    dataset = "ETTm1"   # Weather, ECL, Traffic, ETTh1, ETTh2, ETTm1, ETTm2
    L = 2000
    B = 32
    device = torch.device('cuda:0')   
        
    datapath = "./datasets"
    savepath = f"./Figures/sliding_{dataset}"
   
    df = pd.read_csv(f"{datapath}/{dataset}.csv") 
       
    train_data = df.iloc[:96 + L, 1:].values
    train_date = df.iloc[:96 + L, 0].values
    predict_datas = df.iloc[int(df.shape[0] * 0.9):, 1:].values
    predict_dates = df.iloc[int(df.shape[0] * 0.9):, 0].values
    
    print(df.shape)

        
    sa = SynchronousAlignment(96, train_data, train_data, train_date)
    
    predict_data = []
    predict_date = predict_dates[:B]
    
    for i in range(B):        
        predict_data.append(predict_datas[i: i + 96])
        
    predict_data = torch.tensor(predict_data)

        
    sa.alignment(predict_data, device)
    sa.verify_date(predict_date)
    sa.draw_sliding(0, 1)



