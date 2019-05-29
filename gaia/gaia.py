import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astroquery.gaia import Gaia
import os

class tools(object):
    
    def __init__(self, df):
        self.df = df
        self.path = os.getcwd()
        self.OC = pd.read_csv(self.path+'\\OC_catalouge.csv') 
        self.df['dist_pc'] = 1000/self.df['parallax']

    # 'CMD','radec', 'lb', mov_clust
    def plot(self, plot_type):  
        if plot_type == 'radec':
            plt.figure(figsize = (12,9))
            plt.xlabel('ra')
            plt.ylabel('dec')
            plt.xlim([self.df['ra'].max() + self.df['ra'].mean()*.05, \
                      self.df['ra'].min() - self.df['ra'].mean()*.05])
            plt.scatter(self.df['ra'].values, self.df['dec'].values, s=10, \
                        edgecolors = 'k', facecolors='none', alpha = .75)
        elif plot_type == 'lb':
            l = coord.Angle(self.df['l'].values*u.degree).wrap_at(180*u.degree)
            b = coord.Angle(self.df['b'].values*u.degree)
            fig = plt.figure(figsize=(12,9))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.scatter(l.radian, b.radian, color = 'k', s = 1, alpha = .02)
        elif plot_type == 'CMD':
            def abs_mag(x,y):
                return x - (5 * (np.log10(y/10)))
            M_g = list(map(abs_mag, self.df['phot_g_mean_mag'].values, self.df['parallax'].values))
            M_rp = list(map(abs_mag, self.df['phot_rp_mean_mag'].values, self.df['parallax'].values))
            M_b = list(map(abs_mag, self.df['phot_bp_mean_mag'].values, self.df['parallax'].values))
            color = list(map(lambda x,y: y-x, M_b,M_rp))
            plt.figure(figsize=(8,8))
            plt.xlabel('(Bp-Rp)')
            plt.ylabel('abs_mag')
            plt.ylim([max(M_g)+1, min(M_g)-1])
            plt.xlim([max(color)+1, min(color)-1])
            plt.scatter(color, M_g, marker='o', s=20, edgecolors = 'k', facecolors='none', alpha = .75)
        elif plot_type == 'mov_clust':
            def mas_deg(x):
                return x/(1000*3600)
            plt.figure(figsize=(10,10))
            for i in range(100):
                array = np.arange(0,10,.1)
                star_ra = [self.df['ra'].iloc[i] + self.df['pmra'].iloc[i]*step for step in array]
                star_dec = [self.df['dec'].iloc[i] + self.df['pmdec'].iloc[i]*step for step in array]
                plt.scatter(star_ra, star_dec, s = 1, c = 'k', alpha = .8)   
            plt.ylabel('dec')
            plt.xlabel('ra')
            
        # automatically titles plot in case of singular cluster
        name = tools(self.df).cross_id()['name']
        if len(name) == 1:
            plt.title(name.iloc[0])
        plt.show()
        
    # given df of extracted cluster, this checks if the cluster 
    # exists in https://heasarc.gsfc.nasa.gov/W3Browse/all/mwsc.html
    def cross_id(self):
        RA = self.df['ra'].mean()
        DEC = self.df['dec'].mean()
        DIST = self.df['dist_pc'].mean()
        OC = self.OC
        identified = OC[(OC['ra'] <= RA+5) & (OC['ra'] > RA-5) & \
                        (OC['dec'] > DEC-5) & (OC['dec'] <= DEC+5) & \
                       (OC['distance'] > DIST-10) & (OC['distance'] <= DIST+10)]
        return identified

class query(object):
    
    def __init__(self, my_query):
        self.my_query = my_query
        self.data = self.query()
        
    # query directly from ESA into a pandas dataframe
    def query(self):
        data = Gaia.launch_job_async(self.my_query, dump_to_file=True)
        return data.get_results().to_pandas()
    
    # prints summarized SQL query
    def describe(self):
        string = self.my_query.split("WHERE")[1].strip().split(" ")
        clean_string = [i for i in string if i != '']
        print('Observations:',len(self.data.values))
        print('-'*30)
        i = 0
        while i <= len(clean_string)-1:
            word = clean_string[i]
            if word == 'and':
                print()
            else:
                clean_word = word.replace('gaia.','')
                print(clean_word, end = ' ')
            i += 1

test_cluster = pd.read_csv(os.path.join(os.path.dirname(__file__),'example_data.csv'))


