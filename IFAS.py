#!/usr/bin/env python
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxx-----------SPECTRAL PARAMETERIZATION & CLASSIFICATION USING ANN------------xxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #

# ------------------------------------------------------------------------------------------------------------------- #
# Import Required Libraries
# ------------------------------------------------------------------------------------------------------------------- #
import os
import re
import glob
import itertools
import numpy as np
import pandas as pd
import seaborn as sn

from astropy.io import ascii
from astropy.convolution import convolve, Gaussian1DKernel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FixedLocator, MultipleLocator
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Global Variables
# ------------------------------------------------------------------------------------------------------------------- #
fwhm_jacob = 4.50
fwhm_cflib = 0.88
fwhm_miles = 2.50
lower_lim = 3600
upper_lim = 7400
wave_norm = 5550
sample_size = 5
sample_size2 = 3
min_val = 500
max_val = 7950
list_wave = np.arange(lower_lim, upper_lim + 1, sample_size)
list_wave2 = np.arange(lower_lim, upper_lim + 1, sample_size2)
list_type = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
dict_type = {'O': 1000, 'B': 2000, 'A': 3000, 'F': 4000, 'G': 5000, 'K': 6000, 'M': 7000}
dict_class = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
list_labels = ['Class O', 'Class A', 'Class B', 'Class F', 'Class G', 'Class K', 'Class M']
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# PATH Of Important Directories
# ------------------------------------------------------------------------------------------------------------------- #
DIR_CURNT = os.getcwd()
DIR_JACOB = "/home/avinash/Dropbox/PyCharm/IFAS/Jacobi/"
DIR_CFLIB = "/home/avinash/Dropbox/PyCharm/IFAS/CFLIB/"
DIR_CFLIB2 = "/home/avinash/Dropbox/PyCharm/IFAS/CFLIB2/"
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Functions For File Handling
# ------------------------------------------------------------------------------------------------------------------- #

def group_similar_files(text_list, common_text, exceptions=''):
    """
    Groups similar files based on the string "common_text". Writes the similar files
    onto the list 'text_list' (only if this string is not empty) and appends the similar
    files to a list 'python_list'.
    Args:
        text_list   : Name of the output text file with names grouped based on the 'common_text'
        common_text : String containing partial name of the files to be grouped
        exceptions  : String containing the partial name of the files that need to be excluded
    Returns:
        list_files  : Python list containing the names of the grouped files
    """
    list_files = glob.glob(common_text)
    if exceptions != '':
        list_exception = exceptions.split(',')
        for file_name in glob.glob(common_text):
            for text in list_exception:
                test = re.search(str(text), file_name)
                if test:
                    try:
                        list_files.remove(file_name)
                    except ValueError:
                        pass

    list_files.sort()
    if len(text_list) != 0:
        with open(str(text_list), "w") as f:
            for index in range(0, len(list_files)):
                f.write(str(list_files[index]) + "\n")

    return list_files


def display_text(text_to_display):
    """
    Displays text mentioned in the string 'text_to_display'
    Args:
        text_to_display : Text to be displayed
    Returns:
        None
    """
    print ("\n" + "# " + "-" * (12 + len(text_to_display)) + " #")
    print ("# " + "-" * 5 + " " + str(text_to_display) + " " + "-" * 5 + " #")
    print ("# " + "-" * (12 + len(text_to_display)) + " #" + "\n")

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Make A List of Files From The Two Catalogs
# ------------------------------------------------------------------------------------------------------------------- #
list_jacob = group_similar_files("", os.path.join(DIR_JACOB, "*.ascii"), exceptions='NewData,TempData')
list_cflib = group_similar_files("", os.path.join(DIR_CFLIB, "*.txt"), exceptions='NewData,TempData')
list_cflib2 = group_similar_files("", os.path.join(DIR_CFLIB2, "*.txt"), exceptions='NewData,TempData')
list_cflib2 = [DIR_CFLIB + file_name.split('/')[-1] for file_name in list_cflib2]
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Jacobi - Resample At 5 A, Normalise At 5550 A And Trim From 3600-7400 A
# ------------------------------------------------------------------------------------------------------------------- #
for file_name in list_jacob:
    input_df = pd.read_csv(file_name, names=['Wave', 'Flux'], header=None, sep='\s+', comment='#', engine='python')
    spline = CubicSpline(input_df['Wave'].tolist(), input_df['Flux'].tolist())

    output_df = pd.DataFrame(spline(list_wave), index=list_wave, columns=['Flux'])
    output_df.index.name = 'Wave'
    output_df = output_df.reset_index(drop=False)
    
    norm_factor = output_df[output_df['Wave'] == wave_norm]['Flux']
    output_df['FluxNorm'] = output_df['Flux'].apply(lambda x: x / norm_factor)
    output_df.to_csv(DIR_JACOB + 'NewData_' + file_name.split('/')[-1], sep=' ', index=None, header=True)

display_text("Data From JACOBI Catalog Has Been Pre-Processed")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# CFLIB - Convolve The Spectra With A Gaussian, Resample At 5 A, Normalise At 5550 A And Trim From 3600-7400 A
# ------------------------------------------------------------------------------------------------------------------- #
fwhm_convolve = (fwhm_jacob ** 2 - fwhm_cflib ** 2) ** 0.5
gauss = Gaussian1DKernel(fwhm_convolve)

for file_name in list_cflib:
    input_df = pd.read_csv(file_name, names=['Wave', 'Flux'], header=None, sep='\s+', comment='#', engine='python')
    input_df['FluxCon'] = convolve(input_df['Flux'].tolist(), gauss)
    # input_df.to_csv(DIR_CFLIB + 'TempData_' + file_name.split('/')[-1], sep=' ', index=None, header=True)
    
    spline = CubicSpline(input_df['Wave'].tolist(), input_df['FluxCon'].tolist())
    output_df = pd.DataFrame(spline(list_wave), index=list_wave, columns=['Flux'])
    output_df.index.name = 'Wave'
    output_df = output_df.reset_index(drop=False)
    
    norm_factor = output_df[output_df['Wave'] == wave_norm]['Flux']
    output_df['FluxNorm'] = output_df['Flux'].apply(lambda x: x / norm_factor)
    output_df.to_csv(DIR_CFLIB + 'NewData_' + file_name.split('/')[-1], sep=' ', index=None, header=True)
    
display_text("Data From CFLIB Catalog Has Been Pre-Processed")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# CFLIB2 - Convolve The Spectra With A Gaussian, Resample At 3 A, Normalise At 5550 A And Trim From 3600-7400 A
# ------------------------------------------------------------------------------------------------------------------- #
fwhm_convolve2 = (fwhm_miles ** 2 - fwhm_cflib ** 2) ** 0.5
gauss2 = Gaussian1DKernel(fwhm_convolve2)

for file_name in list_cflib2:
    input_df = pd.read_csv(file_name, names=['Wave', 'Flux'], header=None, sep='\s+', comment='#', engine='python')
    input_df['FluxCon'] = convolve(input_df['Flux'].tolist(), gauss2)
    # input_df.to_csv(DIR_CFLIB2 + 'TempData2_' + file_name.split('/')[-1], sep=' ', index=None, header=True)
    
    spline = CubicSpline(input_df['Wave'].tolist(), input_df['FluxCon'].tolist())
    output_df = pd.DataFrame(spline(list_wave2), index=list_wave2, columns=['Flux'])
    output_df.index.name = 'Wave'
    output_df = output_df.reset_index(drop=False)
    
    norm_factor = output_df[output_df['Wave'] == wave_norm]['Flux']
    output_df['FluxNorm'] = output_df['Flux'].apply(lambda x: x / norm_factor)
    output_df.to_csv(DIR_CFLIB2 + 'NewData2_' + file_name.split('/')[-1], sep=' ', index=None, header=True)
    
display_text("Data From CFLIB2 Catalog Has Been Pre-Processed")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Log And Encode The Spectral Types
# ------------------------------------------------------------------------------------------------------------------- #

def encode_spectype(spectype):
    if re.findall(r'\d+.\d+', spectype):
        subclass = re.findall(r'\d+.\d+', spectype)[0]
    else:
        subclass = re.findall(r'\d+', spectype)[0]
    return dict_type[spectype[0]] + float(subclass) * 100 + 2 * (dict_class[spectype[len(subclass) + 1:]] - 1)

dict_spectype = {}
for file_name in list_cflib:
    with open(file_name, 'r') as fin:
        for line in itertools.islice(fin, 25):
            if re.search('PICKTYPE', line):
                dict_spectype[file_name.split('/')[-1]] = line.rstrip().split(' ')[-1]

data_cflib = pd.DataFrame(dict_spectype.values(), index=dict_spectype.keys(), columns=['TYPE'])
data_cflib.index.name = 'NAME'
data_cflib = data_cflib.reset_index(drop=False)
data_cflib = data_cflib.sort_values(by='NAME')
data_cflib['ENC'] = data_cflib['TYPE'].apply(lambda x: encode_spectype(x))
data_cflib.to_csv('Log_CFLIB.dat', sep=' ', index=False, header=False)
data_cflib['ENC'].to_csv('Type_CFLIB.dat', sep=' ', index=False, header=False)

data_jacob = pd.read_csv('Jacobi_raw.dat', sep='\s+', engine='python')
data_jacob['NAME'] = data_jacob['NAME'].apply(lambda x: x + '.ascii')
data_jacob = data_jacob.sort_values(by='NAME')
data_jacob['ENC'] = data_jacob['TYPE'].apply(lambda x: encode_spectype(x))
data_jacob.to_csv('Log_Jacobi.dat', sep=' ', index=False, header=False)
data_jacob['ENC'].to_csv('Type_Jacobi.dat', sep=' ', index=False, header=False)

data_cflib2 = pd.read_csv('CFLIB2_Params.dat', sep='|', comment='#', engine='python')
data_cflib2['Name'] = data_cflib2['Name'].apply(lambda x: x if x[0:2] != 'HD' else x[2:])
data_cflib2['Name'] = data_cflib2['Name'].apply(lambda x: x.lstrip('0').rstrip())
data_cflib2['Name'] = data_cflib2['Name'].apply(lambda x: DIR_CFLIB + x + '.txt')
data_cflib2 = data_cflib2.sort_values(by='Name')
data_cflib2 = data_cflib2[data_cflib2['Name'].isin(list_cflib2)]
data_cflib2 = data_cflib2[['Teff', 'log(g)', '[Fe/H]']].astype('float64')
data_cflib2['log(g)'] = data_cflib2['log(g)'] * 100
data_cflib2['[Fe/H]'] = data_cflib2['[Fe/H]'] * 100
data_cflib2.to_csv('Param_CFLIB2.dat', sep=' ', index=False, header=False)

data_miles = pd.read_csv('Miles_Params.dat', sep='\s+', header=None, engine='python')
data_miles = data_miles[[2, 3, 4]]
data_miles[3] = data_miles[3] * 100
data_miles[4] = data_miles[4] * 100
data_miles.to_csv('Param_Miles2.dat', sep=' ', index=False, header=False)

display_text("Spectral Types Of Stars In Jacobi, CFLIB & CFLIB2 Have Been Cataloged")
display_text("Spectral Types Have Been Encoded For Stars In Jacobi And CFLIB Catalogs")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Outputs Flux Information From Jacobi & CFLIB Catalog Onto Individual Data Files
# ------------------------------------------------------------------------------------------------------------------- #
list_newjacob = group_similar_files("", os.path.join(DIR_JACOB, "NewData_*.ascii"))
list_newcflib = group_similar_files("", os.path.join(DIR_CFLIB, "NewData_*.txt"))
list_newcflib2 = group_similar_files("", os.path.join(DIR_CFLIB2, "NewData2_*.txt"))

master_jacob = pd.DataFrame()
for file_name in list_newjacob:
    temp_data = pd.read_csv(file_name, sep='\s+', engine='python')
    master_jacob = pd.concat([master_jacob, temp_data['FluxNorm']], axis=1)
master_jacob.T.to_csv('Flux_Jacobi.dat', sep=' ', index=False, header=False)

master_cflib = pd.DataFrame()
for file_name in list_newcflib:
    temp_data = pd.read_csv(file_name, sep='\s+', engine='python')
    master_cflib = pd.concat([master_cflib, temp_data['FluxNorm']], axis=1)
master_cflib.T.to_csv('Flux_CFLIB.dat', sep=' ', index=False, header=False)

master_cflib2 = pd.DataFrame()
for file_name in list_newcflib2:
    temp_data = pd.read_csv(file_name, sep='\s+', engine='python')
    master_cflib2 = pd.concat([master_cflib2, temp_data['FluxNorm']], axis=1)
master_cflib2.T.to_csv('Flux_CFLIB2.dat', sep=' ', index=False, header=False)

display_text("Complete Flux Data From Jacobi, CFLIB & CFLIB2 Cataloged Into Individual Master Output Files")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Read Training And Testing Data Respectively From Flux*.dat and Type*.dat Files
# ------------------------------------------------------------------------------------------------------------------- #
flux_train = pd.read_csv('Flux_Jacobi.dat', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
flux_test = pd.read_csv('Flux_CFLIBx.dat', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
type_train = pd.read_csv('Type_Jacobi.dat', header=None, sep='\s+', dtype='float64', engine='python').T.as_matrix()[0]
type_test = pd.read_csv('Type_CFLIBx.dat', header=None, sep='\s+', dtype='float64', engine='python').T.as_matrix()[0]
type_test = (type_test - 1.5).astype('int')

flux_train2 = pd.read_csv('Flux_Miles2.dat', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
flux_test2 = pd.read_csv('Flux_CFLIB2.dat', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
param_train2 = pd.read_csv('Param_Miles2.dat', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
param_test2 = pd.read_csv('Param_CFLIB2.dat', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plot Histograms Of The Training & Testing Datasets
# ------------------------------------------------------------------------------------------------------------------- #

def plot_hist(file_type, title, plot_name):

    type_df = pd.read_csv(file_type, sep='\s+', names=['NAME', 'TYPE', 'ENC'], header=None, engine='python')
    
    numbers, bins = np.histogram(type_df['ENC'], bins=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    width = 0.9 * (bins[1] - bins[0])
    centers = (bins[:-1] + bins[1:]) / 2
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.bar(centers, numbers, align='center', width=width)
    ax.set_xlim(1000, 8000)
    ax.set_ylabel('Number of Stars', fontsize=14)
    ax.set_xlabel('Spectral Types', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xticks(centers)
    ax.set_xticklabels(list_labels)
    ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

    fig.savefig(plot_name, format='eps', dpi=4000)
    plt.show()
    plt.close(fig)


def plot_hist2(info, title, plot_name):

    numbers, bins = np.histogram(info, bins=np.arange(3000, 43001, 4000))
    width = 0.9 * (bins[1] - bins[0])
    centers = (bins[:-1] + bins[1:]) / 2
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.bar(centers, numbers, align='center', width=width)
    ax.set_ylabel('Number of Stars', fontsize=14)
    ax.set_xlabel('Temperature (In Kelvin)', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xticks(centers)
    ax.tick_params(which='both', direction='in', width=0.8, labelsize=12)

    fig.savefig(plot_name, format='eps', dpi=4000)
    plt.show()
    plt.close(fig)
    
    
def plot_hist3(info, title, plot_name):

    numbers, bins = np.histogram(info, bins=np.arange(10, 100, 10))
    width = 0.9 * (bins[1] - bins[0])
    centers = (bins[:-1] + bins[1:]) / 2
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.bar(centers, numbers, align='center', width=width)
    ax.set_ylabel('Number of Stars', fontsize=14)
    ax.set_xlabel('Log(g)', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xticks(centers)
    ax.set_xticklabels(centers / 100.)
    ax.tick_params(which='both', direction='in', width=0.8, labelsize=12)

    fig.savefig(plot_name, format='eps', dpi=4000)
    plt.show()
    plt.close(fig)
    
    
def plot_hist4(info, title, plot_name):

    numbers, bins = np.histogram(info, bins=np.arange(-300, 100, 40))
    width = 0.9 * (bins[1] - bins[0])
    centers = (bins[:-1] + bins[1:]) / 2
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.bar(centers, numbers, align='center', width=width)
    ax.set_ylabel('Number of Stars', fontsize=14)
    ax.set_xlabel('[Fe/H]', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xticks(centers)
    ax.set_xticklabels(centers / 100.)
    ax.tick_params(which='both', direction='in', width=0.8, labelsize=12)

    fig.savefig(plot_name, format='eps', dpi=4000)
    plt.show()
    plt.close(fig)

plot_hist('Log_Jacobi.dat', title='Training Dataset', plot_name='Plot_TrainingData.eps')
plot_hist('Log_CFLIB.dat', title='Testing Dataset', plot_name='Plot_TestingData.eps')

plot_hist2(param_train2.T[0], title='Training Dataset', plot_name='Plot_Training2Data.eps')
plot_hist2(param_test2.T[0],  title='Testing Dataset', plot_name='Plot_Testing2Data.eps')

plot_hist3(param_train2.T[1], title='Training Dataset', plot_name='Plot_Training3Data.eps')
plot_hist3(param_test2.T[1],  title='Testing Dataset', plot_name='Plot_Testing3Data.eps')

plot_hist4(param_train2.T[2], title='Training Dataset', plot_name='Plot_Training4Data.eps')
plot_hist4(param_test2.T[2],  title='Testing Dataset', plot_name='Plot_Testing4Data.eps')
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plot The Training & Testing Data In Parameter Space
# ------------------------------------------------------------------------------------------------------------------- #

fig = plt.figure(figsize=(16, 8))
cm = plt.cm.get_cmap('RdYlBu')

ax = fig.add_subplot(121)
ax.scatter(param_train2.T[0], param_train2.T[2] / 100, c=param_train2.T[1], cmap=cm)
ax.set_xlabel('Teff (K)', fontsize=16)
ax.set_ylabel('[Fe/H]', fontsize=16)
ax.set_title('Training Data (MILES)', fontsize=16)
ax.tick_params(which='both', direction='in', labelsize=14)
ax.set_xlim(43000, 1000)

ax2 = fig.add_subplot(122, sharey=ax)
ax2.scatter(param_test2.T[0], param_test2.T[2] / 100, c=param_test2.T[1], cmap=cm)
ax2.set_xlabel('Teff (K)', fontsize=16)
ax2.set_ylabel('[Fe/H]', fontsize=16)
ax2.set_title('Testing Data (CFLIB)', fontsize=16)
ax2.tick_params(which='both', direction='in', labelsize=14)
ax2.set_xlim(43000, 1000)

fig.savefig('Plot_ParameterSpace.eps', format='eps', dpi=4000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plot Spectra Of Stars From The Testing Dataset
# ------------------------------------------------------------------------------------------------------------------- #

def plot_cflib(file_plot, path=DIR_CFLIB):
    data_df = pd.read_csv(path + 'TempData_' + file_plot, sep='\s+', engine='python')
    newdata_df = pd.read_csv(path + 'NewData_' + file_plot,  sep='\s+', engine='python')

    spline = CubicSpline(data_df['Wave'].tolist(), data_df['FluxCon'].tolist())
    ax.plot(data_df['Wave'], data_df['Flux'], label='Original Spectra')
    ax.plot(data_df['Wave'], data_df['FluxCon'], label='Convolved Spectra')
    ax.plot(list_wave, spline(list_wave), label='Resampled Spectra')
    ax.plot(newdata_df['Wave'], newdata_df['FluxNorm'], label='Normalised Spectra')

    ax.legend(fontsize=24)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.set_xlabel('Wavelength (In Angstroms)', fontsize=24)
    ax.set_ylabel('Normalised Flux', fontsize=24)
    ax.set_title(file_plot.split('.')[0], fontsize=24)
    ax.tick_params(which='both', direction='in', width=0.3, labelsize=24)

fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
plot_cflib('149757.txt')
ax.set_ylim(1.7, 2.2)
ax.set_xlim(4645, 4700)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
fig.savefig("Plot_PreProcess1.eps", format='eps', dpi=4000)
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
plot_cflib('100889.txt')
ax.set_ylim(0.1, 2.7)
ax.set_xlim(3500, 7500)
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
fig.savefig("Plot_PreProcess2.eps", format='eps', dpi=4000)
plt.show()
plt.close(fig)
    
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plot Spectra Of Stars From The Training Dataset
# ------------------------------------------------------------------------------------------------------------------- #

def plot_jacob(file_plot, offset=0, path=DIR_JACOB):
    data_df = pd.read_csv(path + 'NewData_' + file_plot, sep='\s+', comment='#', engine='python')
    type_df = pd.read_csv('Log_Jacobi.dat', sep='\s+', names=['NAME', 'TYPE', 'ENC'], header=None, engine='python')
    label = type_df.loc[type_df['NAME'] == file_plot, 'TYPE'].item()
    ax.plot(data_df['Wave'], data_df['FluxNorm'] + offset, label=label[0] + ' Type')
    
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)

plot_jacob('JC_1.ascii', offset=0.75)
plot_jacob('JC_17.ascii', offset=0)
plot_jacob('JC_31.ascii', offset=-0.75)
plot_jacob('JC_53.ascii', offset=-1.75)

ax.legend(fontsize=24)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.set_xlabel('Wavelength (In Angstroms)', fontsize=24)
ax.set_ylabel('Normalised Flux', fontsize=24)
ax.tick_params(which='both', direction='in', width=0.3, labelsize=24)

fig.savefig("Plot_SpecType.eps", format='eps', dpi=4000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Training & Testing The Predictions Using The MLP Classifier
# ------------------------------------------------------------------------------------------------------------------- #
mlp = MLPClassifier(hidden_layer_sizes=(66, 66), solver='lbfgs', max_iter=2000, random_state=1,
                    alpha=1e-6, tol=1e-10)
mlp.fit(X=flux_train, y=type_train)

mlp_pred = mlp.predict(X=flux_test)
mlp_dev = type_test - mlp_pred
print(mlp.score(flux_test, type_test))
# print(classification_report(type_test, mlp_pred))

display_text("Training & Testing Done Using MLP Classifier")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Training & Testing The Predictions Using The KNeighbors Classifier
# ------------------------------------------------------------------------------------------------------------------- #
knc = KNeighborsClassifier(n_neighbors=5, leaf_size=30, weights='distance', algorithm='kd_tree')
knc.fit(X=flux_train, y=type_train)

knc_pred = knc.predict(X=flux_test)
knc_dev = type_test - knc_pred
print(knc.score(flux_test, type_test))
# print(classification_report(type_test, knc_pred))

display_text("Training & Testing Done Using KNeighbors Classifier")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Training & Testing The Predictions Using The SVM-NuSVC Classifier
# ------------------------------------------------------------------------------------------------------------------- #
clf = svm.NuSVC(nu=0.42, kernel='sigmoid', coef0=0.019, tol=1e-5)
clf.fit(X=flux_train, y=type_train)

clf_pred = clf.predict(X=flux_test)
clf_dev = type_test - clf_pred
print(knc.score(flux_test, type_test))
# # print(classification_report(type_test, clf_pred))

display_text("Training & Testing Done Using SVM-NuSVC Classifier")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Comparing Different Classifiers For Classification
# ------------------------------------------------------------------------------------------------------------------- #
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), sharey=True)

knc = KNeighborsClassifier(n_neighbors=6, leaf_size=30, weights='distance')
knc.fit(X=flux_train, y=type_train)
ax1.scatter(type_test, knc.predict(flux_test))
ax1.plot(np.arange(min_val, max_val, 5), np.arange(min_val, max_val, 5), linestyle='--', color='k')

ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)
ax1.set_ylabel('ANN', fontsize=14)
ax1.set_xlabel('CFLIB Data', fontsize=14)
ax1.set_title("Nearest Neighbors Classifier", y=1.05, fontsize=14)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.xaxis.set_major_locator(FixedLocator(np.arange(1000, max_val, 1000)))
ax1.xaxis.set_minor_locator(MultipleLocator(250))
ax1.yaxis.set_major_locator(FixedLocator(np.arange(1000, max_val, 1000)))
ax1.yaxis.set_minor_locator(MultipleLocator(250))
ax1.tick_params(which='both', direction='in', width=0.8, labelsize=14)

ax1_opp = ax1.twiny()
ax1_opp.set_xlim(ax1.get_xlim())
ax1_opp.set_xticks(ax1.get_xticks())
ax1_opp.set_xticklabels(list_type)
ax1_opp.tick_params(which='both', direction='in', width=0.8, labelsize=14)
    
mlp = MLPClassifier(hidden_layer_sizes=(64, 64), solver='lbfgs', max_iter=2000, random_state=1, alpha=1e-5)
mlp.fit(X=flux_train, y=type_train)
ax2.scatter(type_test, mlp.predict(flux_test))
ax2.plot(np.arange(min_val, max_val, 5), np.arange(min_val, max_val, 5), linestyle='--', color='k')

ax2.set_xlim(min_val, max_val)
ax2.set_ylim(min_val, max_val)
ax2.set_xlabel('CFLIB Data', fontsize=14)
ax2.set_title("MLP Classifier", y=1.05, fontsize=14)
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_major_locator(FixedLocator(np.arange(1000, max_val, 1000)))
ax2.xaxis.set_minor_locator(MultipleLocator(250))
ax2.yaxis.set_major_locator(FixedLocator(np.arange(1000, max_val, 1000)))
ax2.yaxis.set_minor_locator(MultipleLocator(250))
ax2.tick_params(which='both', direction='in', width=0.8, labelsize=14)

ax2_opp = ax2.twiny()
ax2_opp.set_xlim(ax2.get_xlim())
ax2_opp.set_xticks(ax2.get_xticks())
ax2_opp.set_xticklabels(list_type)
ax2_opp.tick_params(which='both', direction='in', width=0.8, labelsize=14)

plt.subplots_adjust(wspace=0.01)
fig.savefig("Plot_CompClassifiers.eps", format='eps', dpi=2000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Testing Combination Of Different Parameters For MLP Classifiers
# ------------------------------------------------------------------------------------------------------------------- #
mlp_params = [{'solver': 'lbfgs', 'learning_rate': 'constant', 'momentum': 0, 'alpha': 1e-5},
              {'solver': 'lbfgs', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': False, 'alpha': 1e-5},
              {'solver': 'lbfgs', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': True, 'alpha': 1e-5},
              {'solver': 'lbfgs', 'learning_rate': 'invscaling', 'momentum': 0,
               'learning_rate_init': 0.2, 'alpha': 1e-5},
              {'solver': 'lbfgs', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': True, 'alpha': 1e-5},
              {'solver': 'lbfgs', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': False, 'alpha': 1e-5},
              {'solver': 'adam', 'learning_rate_init': 0.01, 'alpha': 1e-5}]

mlp_labels = ["LBFGS, Constant, No Momentum", "LBFGS, Constant, With Momentum",
              "LBFGS, Constant, Nesterov's Momentum", "LBFGS, Inv-scaling, No Momentum",
              "LBFGS, Inv-scaling, With Momentum", "LBFGS, Inv-scaling, Nesterov's Momentum", "Adam"]

mlp_plotargs = [{'c': 'red', 'linestyle': '-'},
                {'c': 'green', 'linestyle': '-'},
                {'c': 'blue', 'linestyle': '-'},
                {'c': 'red', 'linestyle': '--'},
                {'c': 'green', 'linestyle': '--'},
                {'c': 'blue', 'linestyle': '--'},
                {'c': 'black', 'linestyle': '-'}]

list_mlppred = []
for label, param in zip(mlp_labels, mlp_params):
    display_text("Training: {0}".format(label))
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), random_state=1, max_iter=2000, **param)
    mlp.fit(flux_train, type_train)
    list_mlppred.append(mlp.predict(X=flux_test))
    print("Training Set Score: {0}".format(mlp.score(flux_test, type_test)))
    print("Training Set Loss: {0}".format(mlp.loss_))

fig = plt.figure(figsize=(8, 6))

for type_pred, label, args in zip(list_mlppred, mlp_labels, mlp_plotargs):
    ax = fig.add_subplot(111)
    ax.scatter(type_pred, type_test, label=label, **args)
ax.plot(np.arange(min_val, max_val, 5), np.arange(min_val, max_val, 5), linestyle='--', color='k')

ax.legend(fontsize=9)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_ylabel('ANN', fontsize=12)
ax.set_xlabel('CFLIB Data', fontsize=12)
ax.set_title("MLP Classifier", y=1.05, fontsize=14)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_major_locator(MultipleLocator(1000))
ax.xaxis.set_minor_locator(MultipleLocator(250))
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.yaxis.set_minor_locator(MultipleLocator(250))
ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

ax_opp = ax.twiny()
ax_opp.set_xlim(ax.get_xlim())
ax_opp.set_xticks(ax.get_xticks())
ax_opp.set_xticklabels(list_type)
ax_opp.tick_params(which='both', direction='in', width=0.8, labelsize=14)

fig.savefig("Plot_MLPCompare.eps", format='eps', dpi=2000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Testing Combination Of Different Parameters For KNeighbors Classifier
# ------------------------------------------------------------------------------------------------------------------- #
knc_params = [{'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'brute', 'leaf_size': 30, 'p': 2},
              {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'ball_tree', 'leaf_size': 100, 'p': 1},
              {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'kd_tree', 'leaf_size': 100, 'p': 1},
              {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'brute', 'leaf_size': 30, 'p': 2},
              {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'kd_tree', 'leaf_size': 50, 'p': 2},
              {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'kd_tree', 'leaf_size': 50, 'p': 2}]

knc_labels = ["Neighbors 6, Uniform, Brute, 30", "Neighbors 6, Uniform, Ball Tree, 30",
              "Neighbors 6, Distance, KD Tree, 30", "Neighbors 6, Distance, Brute, 30",
              "Neighbors 6, Distance, Ball Tree, 50", "Neighbors 6, Distance, KD Tree, 50"]

knc_plotargs = [{'c': 'red', 'linestyle': '-'}, {'c': 'green', 'linestyle': '-'},
                {'c': 'blue', 'linestyle': '-'}, {'c': 'green', 'linestyle': '--'},
                {'c': 'yellow', 'linestyle': '--'}, {'c': 'black', 'linestyle': '-'}]

list_kncpred = []
for label, param in zip(knc_labels, knc_params):
    display_text("Training: {0}".format(label))
    knc = KNeighborsClassifier(**param)
    knc.fit(flux_train, type_train)
    list_kncpred.append(knc.predict(X=flux_test))
    print("Training Set Score: {0}".format(knc.score(flux_test, type_test)))

fig = plt.figure(figsize=(8, 6))

for type_pred, label, args in zip(list_kncpred, knc_labels, knc_plotargs):
    ax = fig.add_subplot(111)
    ax.scatter(type_pred, type_test, label=label, **args)
ax.plot(np.arange(min_val, max_val, 5), np.arange(min_val, max_val, 5), linestyle='--', color='k')

ax.legend(fontsize=9)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_ylabel('ANN', fontsize=12)
ax.set_xlabel('CFLIB Data', fontsize=12)
ax.set_title("Nearest Neighbors Classifier", y=1.05, fontsize=14)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_major_locator(MultipleLocator(1000))
ax.xaxis.set_minor_locator(MultipleLocator(250))
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.yaxis.set_minor_locator(MultipleLocator(250))
ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

ax_opp = ax.twiny()
ax_opp.set_xlim(ax.get_xlim())
ax_opp.set_xticks(ax.get_xticks())
ax_opp.set_xticklabels(list_type)
ax_opp.tick_params(which='both', direction='in', width=0.8, labelsize=14)

fig.savefig("Plot_KNCompare.eps", format='eps', dpi=2000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Creating The Confusion Matrix For The KNeighbors & MLP Classifier
# ------------------------------------------------------------------------------------------------------------------- #

def create_conf(type_pred):
    conf_df = pd.DataFrame(type_test, columns=['Original'])
    conf_df['Predicted'] = type_pred
    conf_df = conf_df.sort_values(by='Original')

    conf_matrix = np.zeros([7, 7])
    for i in range(0, 7):
        for j in range(0, 7):
            temp_df = conf_df[(conf_df['Original'] > (i + 1) * 1000) & (conf_df['Original'] < (i + 2) * 1000) &
                              (conf_df['Predicted'] > (j + 1) * 1000) & (conf_df['Predicted'] < (j + 2) * 1000)]
            conf_matrix[j, i] = temp_df.shape[0]
            
    conffine_matrix = np.zeros([70, 70])
    for i in range(0, 70):
        for j in range(0, 70):
            temp_df = conf_df[(conf_df['Original'] > 1000 + i * 100) & (conf_df['Original'] < 1000 + (i + 1) * 100) &
                              (conf_df['Predicted'] > 1000 + j * 100) & (conf_df['Predicted'] < 1000 + (j + 1) * 100)]
            conffine_matrix[j, i] = temp_df.shape[0]

    return conf_matrix, conffine_matrix

mlp_conf, mlp_conffine = create_conf(type_pred=mlp_pred)
knc_conf, knc_conffine = create_conf(type_pred=knc_pred)
clf_conf, clf_conffine = create_conf(type_pred=clf_pred)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plot The Fit To The Testing Dataset For Different Classifiers
# ------------------------------------------------------------------------------------------------------------------- #

def plot_fit(type_pred, title, plot_name):
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111)
    ax.scatter(type_test, type_pred)
    ax.plot(np.arange(min_val, max_val, 5), np.arange(min_val, max_val, 5), linestyle='--', color='k')

    ax.legend(fontsize=9)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel('ANN', fontsize=12)
    ax.set_xlabel('CFLIB Data', fontsize=12)
    ax.set_title(title, y=1.05, fontsize=14)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_major_locator(FixedLocator(np.arange(1000, max_val, 1000)))
    ax.xaxis.set_minor_locator(MultipleLocator(250))
    ax.yaxis.set_major_locator(FixedLocator(np.arange(1000, max_val, 1000)))
    ax.yaxis.set_minor_locator(MultipleLocator(250))
    ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

    ax_opp = ax.twiny()
    ax_opp.set_xlim(ax.get_xlim())
    ax_opp.set_xticks(ax.get_xticks())
    ax_opp.set_xticklabels(list_type)
    ax_opp.tick_params(which='both', direction='in', width=0.8, labelsize=14)

    fig.savefig(plot_name, format='eps', dpi=2000)
    plt.show()
    plt.close(fig)

plot_fit(mlp_pred, title='MLP Classifier', plot_name='Plot_MLPFit.eps')
plot_fit(knc_pred, title='KNeighbors Classifier', plot_name='Plot_KNCFit.eps')
plot_fit(clf_pred, title='SVM-NuSVC Classifier', plot_name='Plot_SVMFit.eps')
# ------------------------------------------------------------------------------------------------------------------- #
    
    
# ------------------------------------------------------------------------------------------------------------------- #
# Plot The Deviation Histogram For Different Classifiers
# ------------------------------------------------------------------------------------------------------------------- #

def plot_dev(type_dev, title, plot_name, bins='sqrt'):
    def gaussian_func(x, mean, sigma, const):
        return const * np.exp(-(x - mean) ** 2 / (sigma ** 2))

    mean, sigma = norm.fit(type_dev)
    numbers, bins = np.histogram(type_dev, bins=bins)
    width = 0.9 * (bins[1] - bins[0])
    centers = (bins[:-1] + bins[1:]) / 2
    smooth_centers = np.linspace(min(centers), max(centers), 1000)
    popt, pcov = curve_fit(gaussian_func, centers, numbers, p0=[mean, sigma, max(numbers)])

    print("Number Of Bins = {0}".format(len(centers)))
    print("Mean = {0}".format(mean))
    print("Sigma = {0}".format(sigma))
    print("\nParameters Of The Gaussian Are:\nMean = {0}\nSigma = {1}\nAmplitude = {2}".format(*popt))
    print("Maximum Bar Size = {0}".format(max(numbers)))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.set_xlim(-1600, 1600)
    ax.bar(centers, numbers, align='center', width=width)
    ax.plot(smooth_centers, gaussian_func(smooth_centers, *popt), color='r')
    ax.text(x=0.5 * ax.get_xlim()[-1], y=0.8 * ax.get_ylim()[-1],
            s=" Mean = {0:.1f}\nSigma = {1:.1f}".format(mean, sigma), fontsize=14)

    ax.set_ylabel('Number of Stars', fontsize=14)
    ax.set_xlabel('Deviation (ANN)', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(125))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

    fig.savefig(plot_name, format='eps', dpi=2000)
    plt.show()
    plt.close(fig)
    
plot_dev(mlp_dev, title='MLP Classifier', plot_name='Plot_MLPDeviation.eps')
plot_dev(knc_dev, title='KNeighbors Classifier', plot_name='Plot_KNCDeviation.eps', bins=50)
plot_dev(clf_dev, title='SVM NuSVC Classifier', plot_name='Plot_SVMDeviation.eps', bins=50)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plot The Heatmap Of The Confusion Matrix For Different Classifiers
# ------------------------------------------------------------------------------------------------------------------- #

def plot_confheat(confarr, title, plot_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    xdim, ydim = confarr.shape
    xarr = range(xdim)
    yarr = range(ydim)

    df_cm = pd.DataFrame(confarr, index=xarr, columns=yarr)
    sn.set(font_scale=1.4)
    cbar = sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})

    cbar.set_title('Number of Stars')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Target Class', y=1.10, fontsize=14)
    ax.set_ylabel('ANN Class', fontsize=14)
    ax.set_xticklabels(list_labels)
    ax.set_yticklabels(list_labels[::-1], rotation=0)

    fig.savefig(plot_name, format='eps')
    plt.show()
    plt.close(fig)
    
plot_confheat(mlp_conf, title='MLP Classifier', plot_name='Plot_MLPConfMatrix.eps')
plot_confheat(knc_conf, title='KNeighbors Classifier', plot_name='Plot_KNCConfMatrix.eps')
plot_confheat(clf_conf, title='SVM NuSVC Classifier', plot_name='Plot_SVMConfMatrix.eps')
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plotting The 3D Confusion Matrix For Different Classifiers
# ------------------------------------------------------------------------------------------------------------------- #

def plot_3dconf(confarr, title, plot_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    xdim, ydim = confarr.shape
    xarr = range(xdim)
    yarr = range(ydim)

    xmesh, ymesh = np.meshgrid(xarr, yarr)
    ax.plot_surface(xmesh, ymesh, confarr, color='r')

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 10
    ax.set_xlabel('Target Class', fontsize=14)
    ax.set_ylabel('ANN Class', fontsize=14, rotation=90)
    ax.set_zlabel('Number of Stars', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xticklabels(list_labels, rotation='90')
    ax.set_yticklabels(list_labels)
    ax.view_init(elev=40, azim=40)
    
    fig.savefig(plot_name, format='eps', dpi=2000)
    plt.show()
    plt.close(fig)

plot_3dconf(mlp_conf, title='MLP Classifier', plot_name='Plot_MLP3DConfMatrix.eps')
plot_3dconf(knc_conf, title='KNeighbors Classifier', plot_name='Plot_KNC3DConfMatrix.eps')
plot_3dconf(clf_conf, title='SVM NuSVC Classifier', plot_name='Plot_SVM3DConfMatrix.eps')

plot_3dconf(mlp_conffine, title='MLP Classifier', plot_name='Plot_MLP3DConfFineMatrix.eps')
plot_3dconf(knc_conffine, title='KNeighbors Classifier', plot_name='Plot_KNC3DConfFineMatrix.eps')
plot_3dconf(clf_conffine, title='SVM NuSVC Classifier', plot_name='Plot_SVM3DConfFineMatrix.eps')
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plotting The Contour Of The Confusion Matrix For Different Classifiers
# ------------------------------------------------------------------------------------------------------------------- #

def plot_confcont(confarr, title, plot_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    xdim, ydim = confarr.shape
    xarr = range(xdim)
    yarr = range(ydim)
    
    xmesh, ymesh = np.meshgrid(xarr, yarr)
    ax.contourf(xmesh, ymesh, confarr)

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.set_xlabel('Target Class', fontsize=16)
    ax.set_ylabel('ANN Class', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(list_labels, rotation=90)
    ax.set_yticklabels(list_labels)
    
    fig.savefig(plot_name, format='eps', dpi=2000)
    plt.show()
    plt.close(fig)

plot_confcont(mlp_conf, title='MLP Classifier', plot_name='Plot_MLPConfContour.eps')
plot_confcont(knc_conf, title='KNeighbors Classifier', plot_name='Plot_KNCConfContour.eps')
plot_confcont(clf_conf, title='SVM NuSVC Classifier', plot_name='Plot_SVMConfContour.eps')

plot_confcont(mlp_conffine, title='MLP Classifier', plot_name='Plot_MLP3DConfFineContour.eps')
plot_confcont(knc_conffine, title='KNeighbors Classifier', plot_name='Plot_KNC3DConfFineContour.eps')
plot_confcont(clf_conffine, title='SVM NuSVC Classifier', plot_name='Plot_SVM3DConfFineContour.eps')
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Parameterization Of Teff Using MLP Classifier
# ------------------------------------------------------------------------------------------------------------------- #
mlp = MLPClassifier(hidden_layer_sizes=(75, 75), activation='relu', solver='lbfgs', max_iter=10000, random_state=1,
                    alpha=1e-6, tol=1e-30)
mlp.fit(X=flux_train2, y=param_train2.T[0])
temp_pred = mlp.predict(X=flux_test2)
temp_dev = param_test2.T[0] - temp_pred

print("Ratioed Differential Mean = {0}". format(100 * np.mean((np.absolute(param_test2.T[0] - temp_pred)) /
                                                              param_test2.T[0])))
print("Ratioed Differential Std = {0}". format(100 * np.std((np.absolute(param_test2.T[0] - temp_pred)) /
                                                            param_test2.T[0])))
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Plot For The Parameterization Of Teff, Log(g) & [Fe/H] From MATLAB Output
# ------------------------------------------------------------------------------------------------------------------- #
ann_logg = pd.read_csv('MATLAB_ANNLogg.txt', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
test_logg = pd.read_csv('MATLAB_TestLogg.txt', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
ann_feh = pd.read_csv('MATLAB_ANNFeH.txt', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
test_feh = pd.read_csv('MATLAB_TestFeH.txt', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
ann_teff = pd.read_csv('MATLAB_ANNTeff.txt', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()
test_teff = pd.read_csv('MATLAB_TestTeff.txt', header=None, sep='\s+', dtype='float64', engine='python').as_matrix()

mean_logg, sigma_logg = norm.fit(test_logg - ann_logg)
mean_feh, sigma_feh = norm.fit(test_feh - ann_feh)

data_df = pd.DataFrame(test_teff, columns=['Original'])
data_df['Predicted'] = ann_teff
data_df['Diff'] = data_df['Original'] - data_df['Predicted']
data_df['RelDiff'] = (data_df['Original'] - data_df['Predicted']) / data_df['Original']
data_df = data_df.sort_values(by='Original')

data1 = data_df[data_df['Original'] <= 6000].copy()
data2 = data_df[(data_df['Original'] > 6000) & (data_df['Original'] <= 10000)].copy()
data3 = data_df[data_df['Original'] > 10000].copy()

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={'height_ratios': [2, 1]})

ax1.set_xlim(2800, 6000)
ax1.set_ylim(2800, 6000)
ax1.yaxis.set_major_locator(MultipleLocator(500))
ax1.scatter(data1['Original'], data1['Predicted'])
ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, linestyle='--', color='k')
ax1.set_ylabel('Teff (ANN, K)', fontsize=20)
ax1.tick_params(which='both', labelsize=16)
ax1.set_xticklabels([])

ax4.set_xlim(2800, 6000)
ax4.set_ylim(-1, 0.5)
ax4.plot([3000, 6000], [0, 0], linestyle='--', color='k')
ax4.scatter(data1['Original'], data1['RelDiff'])
ax4.set_ylabel('Fractional Error', fontsize=20)
ax4.tick_params(which='both', labelsize=16)

ax2.set_xlim(5700, 10000)
ax2.set_ylim(5700, 10000)
ax2.yaxis.set_major_locator(MultipleLocator(1000))
ax2.scatter(data2['Original'], data2['Predicted'])
ax2.plot([0, 1], [0, 1], transform=ax2.transAxes, linestyle='--', color='k')
ax2.tick_params(which='both', labelsize=16)
ax2.set_xticklabels([])

ax5.set_xlim(5700, 10000)
ax5.set_ylim(-0.5, 0.4)
ax5.scatter(data2['Original'], data2['RelDiff'])
ax5.set_xlabel('Teff (Literature, K)', fontsize=20)
ax5.tick_params(which='both', labelsize=16)
ax5.plot([5700, 10000], [0, 0], linestyle='--', color='k')
    
ax3.set_xlim(9000, 43000)
ax3.set_ylim(9000, 43000)
ax3.xaxis.set_major_locator(MultipleLocator(10000))
ax3.yaxis.set_major_locator(MultipleLocator(10000))
ax3.scatter(data3['Original'], data3['Predicted'])
ax3.plot([0, 1], [0, 1], transform=ax3.transAxes, linestyle='--', color='k')
ax3.tick_params(which='both', labelsize=16)
ax3.set_xticklabels([])

ax6.set_xlim(9000, 43000)
ax6.set_ylim(-1.2, 1)
ax6.xaxis.set_major_locator(MultipleLocator(10000))
ax6.scatter(data3['Original'], data3['RelDiff'])
ax6.tick_params(which='both', labelsize=16)
ax6.plot([9000, 43000], [0, 0], linestyle='--', color='k')

fig.subplots_adjust(hspace=0.05, wspace=0.2)
fig.savefig('Plot_MLPTempFitSplit.eps', format='eps', dpi=4000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.set_xlim(1000, 43000)
ax.set_ylim(1000, 43000)
ax.xaxis.set_major_locator(MultipleLocator(5000))
ax.yaxis.set_major_locator(MultipleLocator(5000))
ax.scatter(data_df['Original'], data_df['Predicted'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='k')
ax.set_title('MLP Classifier', fontsize=14)
ax.set_ylabel('Teff (ANN, K)', fontsize=14)
ax.set_xlabel('Teff (Literature, K)', fontsize=14)

fig.savefig('Plot_MLPTempFit.eps', format='eps', dpi=4000)
plt.show()
plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------- #

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.scatter(test_logg, ann_logg)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.text(x=0, y=4, s=" Mean = {0:.2f}\nSigma = {1:.2f}".format(mean_logg, sigma_logg), fontsize=14)

ax.legend(fontsize=9)
ax.set_ylim(-1, 5)
ax.set_xlim(-1, 5)
ax.set_ylabel('log(g) (ANN, dex)', fontsize=16)
ax.set_xlabel('log(g) (Literature, dex)', fontsize=16)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

fig.savefig('Plot_MLPLoggFit.eps', format='eps', dpi=2000)
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.scatter(test_feh, ann_feh)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.text(x=-2.5, y=0.5, s=" Mean = {0:.2f}\nSigma = {1:.2f}".format(mean_feh, sigma_feh), fontsize=14)

ax.legend(fontsize=9)
ax.set_ylim(-3, 1)
ax.set_xlim(-3, 1)
ax.set_ylabel('[Fe/H] (ANN, dex)', fontsize=16)
ax.set_xlabel('[Fe/H] (Literature, dex)', fontsize=16)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

fig.savefig('Plot_MLPFeHFit.eps', format='eps', dpi=2000)
plt.show()
plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Parameterization Of Log(g) Using MLP Classifier
# ------------------------------------------------------------------------------------------------------------------- #
arr_train2 = param_train2.T[1]

mlp = MLPClassifier()
mlp.fit(X=flux_train2, y=arr_train2)

logg_pred = mlp.predict(X=flux_test2)
logg_dev = param_test2.T[1] - logg_pred

mean, sigma = norm.fit(logg_dev)
print("Mean = {0}". format(mean))
print("Sigma = {0}". format(sigma))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.scatter(param_test2.T[1], logg_pred)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)

ax.legend(fontsize=9)
ax.set_xlim(0, 520)
ax.set_ylim(0, 520)
ax.set_ylabel('ANN', fontsize=12)
ax.set_xlabel('CFLIB Data', fontsize=12)
ax.set_title(title, y=1.05, fontsize=14)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)
ax.set_xticklabels(ax.get_xticks() / 100.)
ax.set_yticklabels(ax.get_yticks() / 100.)

fig.savefig(plot_name, format='eps', dpi=2000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Parameterization Of [Fe/H] Using MLP Classifier
# ------------------------------------------------------------------------------------------------------------------- #
arr_train2 = param_train2.T[2]

mlp = MLPClassifier(hidden_layer_sizes=(80, 80), max_iter=500, random_state=1, alpha=1e-6, tol=1e-30)
mlp.fit(X=flux_train2, y=arr_train2)

feh_pred = mlp.predict(X=flux_test2)
feh_dev = param_test2.T[2] - feh_pred

mean, sigma = norm.fit(feh_dev)
print("Mean = {0}". format(mean))
print("Sigma = {0}". format(sigma))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.scatter(param_test2.T[2], feh_pred)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)

ax.legend(fontsize=9)
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_ylabel('ANN', fontsize=12)
ax.set_xlabel('CFLIB Data', fontsize=12)
ax.set_title(title, y=1.05, fontsize=14)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(which='both', direction='in', width=0.8, labelsize=14)

fig.savefig(plot_name, format='eps', dpi=2000)
plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #
