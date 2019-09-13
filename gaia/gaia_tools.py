import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import astropy.coordinates as coord
import astropy.units as u
from astroquery.gaia import Gaia

demo_data = pd.read_csv(os.getcwd()+'\\viz_demo_data.csv')

# basic unit conversions 
# mas = miliarcseconds 
# deg = degrees
# pc = parsecs
# au = astronomical units
class convert:
    def __init__(self, value):
        self.value = value
    def pc_ly(self):
        return 3.26156/(float(self.value)/1000)
    def mas_deg(self):
        return self.value*(1/1000)/3600
    def mas_pc(self):
        return 1/(self.value/1000)
    def pc_mas(self):
        return 1000/self.value
    def au_pc(self):
        return self.value/206264.806

class viz:
    def __init__(self, df):
        self.df = df
        self.path = os.getcwd()
        self.OC = pd.read_csv(self.path+'\\ocgaiacat.csv') 
    
    # calculating absolute magnitude and colors of stars
    def get_CMD(self):
        def abs_mag(x,y):
            return x - (5 * (np.log10(y/10)))
        g = list(map(abs_mag, self.df['phot_g_mean_mag'].values, self.df['parallax'].values))
        rp = list(map(abs_mag, self.df['phot_rp_mean_mag'].values, self.df['parallax'].values))
        b = list(map(abs_mag, self.df['phot_bp_mean_mag'].values, self.df['parallax'].values))
        color = list(map(lambda x,y: y-x, b, rp))
        return g, rp, b, color
    
    # used for normalizing data for CNN, setting 2D with respect to origin
    def normalize(self, *args):
        for x in args:
            mu = np.mean(x)
            x-= mu
            yield x
    
    def plot(self, plot_type):  
        if plot_type == 'radec':
            plt.figure(figsize = (12,9))
            plt.xlabel('ra')
            plt.ylabel('dec')
            plt.xlim([self.df['ra'].max() + self.df['ra'].mean()*.05, \
                      self.df['ra'].min() - self.df['ra'].mean()*.05])
            plt.scatter(self.df['ra'].values, self.df['dec'].values, s=2, \
                        edgecolors = 'k', facecolors='none', alpha = .5)
        elif plot_type == 'lb':
            l = coord.Angle(self.df['l'].values*u.degree).wrap_at(180*u.degree)
            b = coord.Angle(self.df['b'].values*u.degree)
            fig = plt.figure(figsize=(12,9))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.scatter(l.radian, b.radian, color = 'k', s = 1, alpha = .2)
        elif plot_type == 'CMD':
            g, rp, b, color = self.get_CMD()
            plt.figure(figsize=(6,6))
            plt.xlabel('(Bp-Rp)')
            plt.ylabel('abs_mag')
            plt.ylim([max(g)+1, min(g)-1])
            plt.xlim([max(color)+1, min(color)-1])
            plt.scatter(color, g, marker='o', s=20, edgecolors = 'k', facecolors='none', alpha = .75)
        elif plot_type == 'movclust':
            plt.figure(figsize=(10,10))
            for i in range(100):
                array = np.arange(0,10,.1)
                star_ra = [self.df['ra'].iloc[i] + self.df['pmra'].iloc[i]*step for step in array]
                star_dec = [self.df['dec'].iloc[i] + self.df['pmdec'].iloc[i]*step for step in array]
                plt.scatter(star_ra, star_dec, s = 1, c = 'k', alpha = .8)   
            plt.ylabel('dec')
            plt.xlabel('ra')
        elif plot_type == 'heatmap':
            CMD_data = self.get_CMD()
            norm_g, norm_color = self.normalize(CMD_data[0],CMD_data[3])
            plt.figure(figsize=(6,6))
            plt.hist2d(norm_color,norm_g, bins=[np.linspace(-3,3,100),np.linspace(-10,10,100)], cmap = 'inferno')
            plt.xlim([3,-3])
            plt.ylim([10,-10])
            
        # automatically titles plot in case of singular cluster
        name = viz(self.df).cross_id()['name']
        if len(name) == 1:
            plt.title(name.iloc[0])
        plt.show()
        
    # given data of an extracted cluster, check if known cluster exists in this region
    def cross_id(self):
        df = self.df
        OC = self.OC
        identified = OC[(OC['ra'] <= df['ra'].max()) & (OC['ra'] > df['ra'].min()) & \
                        (OC['dec'] > df['dec'].min()) & (OC['dec'] <= df['dec'].max()) & \
                       (OC['parallax'] > df['parallax'].min()) & (OC['parallax'] <= df['parallax'].max())]
        return identified

class query:
    
    def __init__(self):
        self.OC = pd.read_csv(os.getcwd()+'\\ocgaiacat.csv') 

    def archive_pull(self, my_query):
        # https://astroquery.readthedocs.io/en/latest/gaia/gaia.html
        data = Gaia.launch_job_async(my_query, dump_to_file=True)
        return data.get_results().to_pandas()
    
    def get_clusters(self, g, n, N, d, D):
        
        # list of queried clusters
        clusters = []
        
        # subsetting star cluster catalogue for clusters within inputed params
        OC = self.OC[(self.OC['Nstars'] >= n) & 
                     (self.OC['d'] >= d) & 
                     (self.OC['d'] < D)]
        print("{} open cluster(s) in the interval [{},{}] pc.".format(len(OC), d, D))
        
        # for each cluster in OC dataframe, query Gaia satellite for stars in that region
        for index in OC.index:
            clust = OC.loc[index]
            name = clust['name']
            
            # the variables below are measured in degrees and scaled by constant g if desired
            delta = clust['radius']
            min_ra = clust['ra'] - delta*g ; max_ra = clust['ra'] + delta*g
            min_dec = clust['dec'] - delta*g ; max_dec = clust['dec'] + delta*g
            
            # parsecs -> parallax (miliarcseconds)
            rho_max =  1000/(clust['dmin'])
            rho_min = 1000/(clust['dmax'])
            
            template = f"SELECT gaia.source_id, gaia.l, gaia.b, gaia.ra, gaia.dec, \
                        gaia.pmra, gaia.pmra_error, gaia.pmdec, gaia.pmdec_error, \
                        gaia.radial_velocity, gaia.radial_velocity_error, gaia.parallax, \
                        gaia.parallax_error, gaia.phot_g_mean_mag, \
                        gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag \
                        \
                        FROM gaiadr2.gaia_source as gaia \
                        \
                        WHERE gaia.parallax > 0 and gaia.parallax >= {rho_min} and gaia.parallax < {rho_max} \
                        and gaia.ra >= {min_ra} and gaia.ra < {max_ra} \
                        and gaia.dec >= {min_dec} and gaia.dec < {max_dec} \
                        and (gaia.parallax_error/gaia.parallax) < .2 \
                        and gaia.pmra_error < 1.5 and gaia.pmdec_error < 1.5 and \
                        gaia.pmra is not null and gaia.pmdec is not null and gaia.phot_bp_mean_mag is not null and \
                        gaia.phot_rp_mean_mag is not null and gaia.phot_g_mean_mag is not null"
            
            # data queried from https://gea.esac.esa.int/archive/
            pull = self.archive_pull(template)

            if len(pull) > N:
                # mas -> pc
                pull['dist_pc'] = 1000/pull['parallax'] 
                pull['PM'] = 1000/pull['parallax']

                # using z-score to remove any outliers that will affect center of mass calculation
                pull = pull[(pull['PM']-pull['PM'].mean())/pull['PM'].std() < 1.96]
                
                # preparing data for gaussian mixture model
                # goal is to seperate cluster members from field stars
                X = pd.DataFrame(columns = ['pmra','pmdec'])
                X['pmra'] = pull['pmra'] 
                X['pmdec'] = pull['pmdec']
                gmm = GaussianMixture(n_components=2, tol = 1e-6, covariance_type = 'spherical').fit(X)
                labels = gmm.predict(X)
                pull['labels'] = labels
                
                # selecting subpopulation with lowest distance from center of mass
                holder = {i: 0 for i in labels} # holds mean distance and label 
                for label in set(labels):
                    subset = pull[pull['labels'] == label]
                    COM = (subset['pmra'].mean(), subset['pmdec'].mean())
                    dist = np.sqrt((subset['pmra'] - COM[0])**2 + (subset['pmdec'] - COM[1])**2)
                    mu_dist = np.median(dist)

                    # this prevents subsets of 1 or 2 stars from 
                    # skewing selection due to mean = approx 0
                    if len(subset) <= 2:
                        holder[label] = np.inf
                    else:
                        holder[label] = mu_dist

                # dropping field stars in favor of most clustered sub population & dropping excess rows
                cluster_members = pull[pull['labels'] == min(holder, key=holder.get)]

                clusters.append([name, cluster_members])
                
                # DELETE - This is for DEMO
                plt.figure(figsize = (6,6))
                plt.scatter(X['pmra'], X['pmdec'], s=10, c=labels, cmap='winter') 
                plt.show()
                viz(cluster_members).plot('CMD')

        results = []
        for i in clusters:
            df = i[1]
            df['name'] = [i[0] for x in range(len(df))]
            results.append(df)
        if len(results) == 0:
            print("Adjust Search Parameters")
        else:
            return pd.concat(results) 

    def get_fieldstars(self, iterations, lower_n, upper_N, d, D):

        # list of queried clusters
        field_stars = []
        
        def scale_delta(dist):
            return 10/dist
        
        for i in range(iterations):
                
            # base coordinates 
            ra = random.uniform(0,360)
            dec = random.uniform(0,90)
            dist = random.uniform(d,D)
            
            # delta (degrees) is the radius of the spherical partition 
            # scale_delta() scales it based on distance, the further the distance, the smaller
            # step size is desired when increasing the radius 
            # Taking input in parsecs, f(200) = 2 deg f(2000) = .05 degrees || f =- 100/dist
            delta = scale_delta(dist)
            # condition that query return >= N stars 
            N = random.randint(lower_n, upper_N)
            condition = False 
            
            while condition == False:
                 
                # bounds 
                min_ra = ra - delta ; max_ra = ra + delta
                min_dec = dec - delta ; max_dec = dec + delta 
                parallax_scaled =  np.tan(np.radians(delta))*dist
                # converting pc -> mas
                rho_min = 1000/(dist + parallax_scaled)
                rho_max = 1000/(dist - parallax_scaled)
                
                template = f"SELECT gaia.source_id, gaia.l, gaia.b, gaia.ra, gaia.dec, \
                    gaia.pmra, gaia.pmra_error, gaia.pmdec, gaia.pmdec_error, \
                    gaia.radial_velocity, gaia.radial_velocity_error, gaia.parallax, \
                    gaia.parallax_error, gaia.phot_g_mean_mag, \
                    gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag \
                    \
                    FROM gaiadr2.gaia_source as gaia \
                    \
                    WHERE gaia.parallax > 0 and gaia.parallax >= {rho_min} and gaia.parallax < {rho_max} \
                    and gaia.ra >= {min_ra} and gaia.ra < {max_ra} \
                    and gaia.dec >= {min_dec} and gaia.dec < {max_dec} \
                    and (gaia.parallax_error/gaia.parallax) < .2 \
                    and gaia.pmra_error < 1.5 and gaia.pmdec_error < 1.5 and \
                    gaia.pmra is not null and gaia.pmdec is not null and gaia.phot_bp_mean_mag is not null and \
                    gaia.phot_rp_mean_mag is not null and gaia.phot_g_mean_mag is not null"
                
                print('Rhomin: {} Rhomax:{}'.format(1000/rho_min,1000/rho_max))
                partition = self.archive_pull(template)
                if len(partition) > 2:
                    viz(partition).plot('CMD')
                if len(partition) >= N:
                    partition['dist_pc'] = 1000/partition['parallax']
                    
                    # note that FS # is naming convention equivalent to cluster having a name (FS = Field Stars)
                    field_stars.append(['FS{}'.format(i), partition])
                    condition = True
                delta += delta
                
                    
        results = []
        for i in field_stars:
            df = i[1]
            df['name'] = [i[0] for x in range(len(df))]
            results.append(df)
        
        return pd.concat(results) 