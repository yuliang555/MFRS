import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from math import ceil, sin, pi
from sklearn.cluster import DBSCAN


class ReferenceSeries:
    def __init__(self, data, rs_type="sin", Q=10, resolution=10, alpha=0.7, pr=False):
        self.L, self.C = data.shape
        self.rs_type = rs_type
        self.Q = Q
        self.alpha = alpha
        self.pr = pr
        
        for n in range(2, self.L):
            if round(self.L / (n - 1)) < round(self.L / n) + resolution:
                break            
        self.boundary = round(self.L / n)
        
        self.frequency, self.fre_magnitude = self._fft(data)
        self.period, self.per_magnitude = self._convert_spectrum()

    
    def _fft(self, data):
        # remove dc component
        mean = np.mean(data)
        data = data - mean

        #  fast fourier transform
        fft_result = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(self.L)
        frequency = freqs[freqs >= 0]
        fre_magnitude = np.abs(fft_result[freqs >= 0])
        
        return frequency, fre_magnitude        

    
    def _convert_spectrum(self):
        period = np.array([i for i in range(self.boundary)])
        per_magnitude = np.zeros((self.boundary, self.C))
        
        for per in period[4:]:
            indices = self.get_indices_by_per(per)
            per_magnitude[per] = np.max(self.fre_magnitude[indices, :], axis=0)
            
        return period, per_magnitude


    def get_indices_by_per(self, pbp, hbp=1):
        index = round(self.L * hbp / pbp)
        
        if round(self.L * hbp / (index - 2)) == pbp:
            start = index - 2
        elif round(self.L * hbp / (index - 2)) == pbp:
            start = index - 1
        else:
            start = index
            
        if round(self.L * hbp / (index + 2)) == pbp:
            end = index + 2
        elif round(self.L * hbp / (index + 1)) == pbp:
            end = index + 1
        else:
            end = index
         
        indices = np.array([n for n in range(start, end + 1)])
        
        return indices

                
    def channel_pbp_extractor(self, magnitude):                
        pbp = []
        
        for per in range(4, self.boundary // 2):
            if magnitude[per] == np.max(magnitude[:per * 2 + 1]):
                pbp.append(per)
                magnitude[:per * 2 + 1] = 0
                
        return pbp

    
    def pbp_extractor(self):
        per_magnitude_copy = deepcopy(self.per_magnitude)
        pbps = []
        all_channels_pbp = []
        
        for channel in range(self.C):
            pbp = self.channel_pbp_extractor(per_magnitude_copy[:, channel])
            if self.pr:
                print(f"{channel}:    {pbp}")            
            all_channels_pbp += pbp
            
        all_channels_pbp = np.array(all_channels_pbp)
        dbscan = DBSCAN(eps=2, min_samples=round(self.C * self.alpha))
        clusters = dbscan.fit_predict(all_channels_pbp.reshape(-1, 1))

        for cluster in set(clusters) - {-1}:
            indices = [index for index in range(len(clusters)) if clusters[index] == cluster]
            cluster_points = all_channels_pbp[indices]
            center = round(np.mean(cluster_points))
            pbps.append(center)           
        
        assert len(pbp) > 0, "primary base-pattern not found, please decrease the value of alpha to relax the conditions and continue extracting"
                        
        self.pbps = np.array(pbps)
        
        if self.pr:
            print(f"cluster pbp:    {pbps}")

        
    def base_pattern_extractor(self, manual_add=[]):
        constraint = 3
        bps_scores = dict(zip([(pbp, 1) for pbp in self.pbps], np.ones(len(self.pbps))))
        base_magnitude = np.max(self.per_magnitude[self.pbps])
        
        for pbp in self.pbps:
            # base_magnitude = self.per_magnitude[pbp]
                        
            # for h in range(2, min(int(pbp / constraint) + 1, self.Q + 1)):
            #     indices = self.get_indices_by_per(pbp, h)
            #     scores = np.mean(np.max(self.fre_magnitude[indices], axis=0) / base_magnitude)            
            #     bps_scores.update(dict(zip([(pbp, h)], [scores]))) 
                           
            harmonic = np.array([h for h in range(2, min(int(pbp / constraint) + 1, self.Q + 2))])
                        
            indices = np.round(self.L * harmonic / pbp).astype(int)
            scores = np.mean(self.fre_magnitude[indices] / base_magnitude, axis=1)
            
            base_patterns = [(pbp, h) for h in harmonic]            
            bps_scores.update(dict(zip(base_patterns, scores)))
            
            constraint = pbp * 2
        
        sorted_bps_scores = sorted(bps_scores.items(), key=lambda item: item[1], reverse=True)
        self.base_patterns = list(dict(sorted_bps_scores[:self.Q + len(self.pbps)]).keys())            
        
        if self.pr:
            print(f"manual setting base-patterns:    {manual_add}")
            print(f"auto extrating base-patterns:    {self.base_patterns}")
            
        self.base_patterns += manual_add
        
        return sorted_bps_scores

    
    def reference_series_generator(self):
        assert self.rs_type in ["sin", "swatooth", "reactangle", "pulse"], "rs_type must be one of sin, swatooth, reactangle, or pulse"
        
        if self.rs_type == "sin":
            rs = [[sin(t * 2 * pi * bp[1] / bp[0]) for t in range(self.L)] for bp in self.base_patterns]
        elif self.rs_type == "swatooth":
            rs = [[t * bp[1] % bp[0] for t in range(self.L)] for bp in self.base_patterns]
        elif self.rs_type == "reactangle":
            rs = [[2 * t * bp[1] // bp[0] % 2 for t in range(self.L)] for bp in self.base_patterns]
        elif self.rs_type == "pulse":
            rs = [[1 if t * bp[1] % bp[0] == 0 else 0 for t in range(self.L)] for bp in self.base_patterns]
        
        return np.array(rs).T  

    
    def draw_channel_spectrum(self, channel):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        pbp = self.pbps[0]
        harmonic = np.array([1, 2])
        indices = np.round(self.L * harmonic / pbp).astype(int)
        xticks = [f"{h}/{pbp}" for h in harmonic]

        index = self.L // 100
        ax[0].plot(self.frequency[:index], self.fre_magnitude[:index, channel])
        # ax[0].scatter(self.frequency[indices], self.fre_magnitude[indices, channel], color="red")
        # ax[0].set_xticks(self.frequency[indices], xticks, fontsize=12, rotation=-45)
        ax[0].set_yticks([])
        ax[0].set_xlabel(r'$f$', fontsize=30)
        ax[0].xaxis.set_label_coords(0.9, -0.04)
        ax[0].set_ylabel('Amplitude', fontsize=20)
        ax[0].set_ylim(0, self.per_magnitude[pbp, channel] * 2)                
        
        xticks = [f"{pbp}" for pbp in self.pbps]
        
        ax[1].plot(self.period, self.per_magnitude[:, channel])
        ax[1].scatter([self.pbps], [self.per_magnitude[self.pbps, channel]], color="r")
        ax[1].set_xticks(self.pbps, xticks, fontsize=12, y=-0.02)
        ax[1].set_yticks([])
        ax[1].set_xlabel(r'$\mathcal{T}$', fontsize=30)
        ax[1].xaxis.set_label_coords(0.9, -0.04)
        ax[1].set_ylabel('Amplitude', fontsize=20)
        
        fig.savefig(f"{savepath}_{channel}.png")
        


if __name__ == "__main__":
    dataset = "ETTm2"   # Solar, Weather, ECL, Traffic, ETTh1, ETTh2, ETTm1, ETTm2
    datapath = "./datasets"
    savepath = f"./Figures/spectrum_{dataset}"


    if dataset == "Solar":
        df_raw = []
        with open("./datasets/Solar.txt", "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df = pd.DataFrame(df_raw)
        data = df.values
    else:
        df = pd.read_csv(f"{datapath}/{dataset}.csv")    
        data = df.iloc[:, 1:].values
    
    print(df.shape)

        
    rs = ReferenceSeries(data, pr=True)
    
    rs.pbp_extractor()
    rs.base_pattern_extractor()    
    rs.draw_channel_spectrum(0)