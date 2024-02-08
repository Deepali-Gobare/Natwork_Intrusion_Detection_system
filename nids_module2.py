import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
import random

# Avoid Printing Warnings

from warnings import filterwarnings
filterwarnings('ignore')

#Column Names
col_names = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

#Loading the Dataset
train = pd.read_csv(r'NSL_KDD_Train (1).csv', header=None,names=col_names)
test = pd.read_csv(r'NSL_KDD_Test (1).csv', header=None,names=col_names)
df=pd.read_csv(r'NSL_KDD_Train (1).csv', header=None,names=col_names)
df_test=pd.read_csv(r'NSL_KDD_Test (1).csv', header=None,names=col_names)

#Checking Data Dimensions
print(train.head())

print(f'Dimensions of the Training set:{df.shape}')
print(f'Dimensions of the Test set:{df_test.shape}')

# #Data Pre-Processing
# #Mapping Normal as 0 and Attack as 1 (Encoding)

# # Train Dataset
is_attack = train.attack.map(lambda a: 0 if a == 'normal' else 1)

# # Test Dataset
test_attack = test.attack.map(lambda a: 0 if a == 'normal' else 1)

# #Adding Column to Actual Dataset
# # Adding to Train Dataset
train['attack_flag'] = is_attack

# # Addings to Test Dataset
test['attack_flag'] = test_attack
print(train.head())

# #Classifying Attacks into 4 Categories
# # There are a lot of different types of attacks provided in the dataset. We will classify the attacks into categories of 4 and then do the classification for these 4 classes.

# # The classification will be as follows:

# # Denial of Service attacks: apache2,back,land, neptune, mailbomb, pod, processtable, smurf, teardrop, udpstorm, worm

# # Probe attacks: ipsweep, mscan, nmap, portsweep, saint, satan

# # Privilege escalation attacks: buffer_overflow, loadmdoule, perl, ps, rootkit, sqlattack, xterm

# # Remote access attacks: ftp_write, guess_passwd, http_tunnel, imap, multihop, named, phf, sendmail, snmpgetattack, snmpguess, spy, warezclient, warezmaster, xclock, xsnoop

# # Hence the attack labels will be - DOS, Probe, Privilege, Access, Normal

# # lists to hold our attack classifications

dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']
# # Attack Labels

# attack_labels = ['Normal','DoS','Probe','Privilege','Access']
# # Mapping Attack Labels to Numbers

def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0
        
    return attack_type
# Mapping Data for Train Dataset

attack_map = train.attack.apply(map_attack)
train['attack_map'] = attack_map
# Mapping Data for Test Dataset

test_attack_map = test.attack.apply(map_attack)
test['attack_map'] = test_attack_map
train.head()

# Data Profiling and Visualization
# Checking Attack Types and the Protocol Counts
attack_vs_protocol = pd.crosstab(train.attack, train.protocol_type)
attack_vs_protocol

# Rendering Pie Charts

def bake_pies(data_list,labels):
    list_length = len(data_list)
    
    # setup for mapping colors
    color_list = sns.color_palette('flare')
    color_cycle = itertools.cycle(color_list)
    cdict = {}
    
    # build the subplots
    fig, axs = plt.subplots(1, list_length,figsize=(18,10), tight_layout=False)
    plt.subplots_adjust(wspace=1/list_length)
    
    # loop through the data sets and build the charts
    for count, data_set in enumerate(data_list): 
        
        # update our color mapt with new values
        for num, value in enumerate(np.unique(data_set.index)):
            if value not in cdict:
                cdict[value] = next(color_cycle)
       
        # build the wedges
        wedges,texts = axs[count].pie(data_set,
                           colors=[cdict[v] for v in data_set.index])

        # build the legend
        axs[count].legend(wedges, data_set.index,
                           title="Flags",
                           loc="center left",
                           bbox_to_anchor=(1, 0, 0.5, 1))
        # set the title
        axs[count].set_title(labels[count])
        
    return axs 

#Normal Attack Class Counts
train['target'] = train.iloc[:,41].apply(lambda x: 0 if x == 'normal' else 1)
test['target'] = test.iloc[:,41].apply(lambda x: 0 if x == 'normal' else 1)
data = train['target'].value_counts()
lab = ['normal','attack']
data

# Plotting Pie Chart

# %matplotlib inline
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(aspect="equal"))
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
ax.legend(wedges, lab,
          title="Label",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=10, weight="bold")

ax.set_title("Distribution of Labels")

plt.show()

#Attack Type Distribution
# Calculating Number of Occurences of Each Type of Attack

DOS = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','upstorm','worm']
Probe = ['ipsweep','nmap','mscan','portsweep','saint','satan']
U2R = ['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm']
R2L = ['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','Snmpgetattack','spy','snmpguess','warzclient','warzmaster','xlock','xsnoop']
count = {'DOS':0, 'Probe':0, 'U2R':0, 'R2L':0}
for attack in train.attack:
    if attack in DOS:
        count['DOS'] += 1
    elif attack in Probe:
        count['Probe'] += 1
    elif attack in U2R:
        count['U2R'] += 1
    elif attack in R2L:
        count['R2L'] += 1
count

#Distribution of Attack Pie Chart
data = list(count.values())
lab = list(count.keys())
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(aspect="equal"))
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
ax.legend(wedges, lab,
          title="Label",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=10, weight="bold")

ax.set_title("Distribution of Attacks")

plt.show()
data

#Protocols and Occurence of Attacks for Each Protocol
# get the series for each protocol

icmp_attacks = attack_vs_protocol.icmp
tcp_attacks = attack_vs_protocol.tcp
udp_attacks = attack_vs_protocol.udp

# create the charts

bake_pies([icmp_attacks, tcp_attacks, udp_attacks],['icmp','tcp','udp'])
plt.show()

#Normal-Attack Attack Type Distribution
# get a series with the count of each flag for attack and normal traffic

normal_flags = train.loc[train.attack_flag == 0].flag.value_counts()
attack_flags = train.loc[train.attack_flag == 1].flag.value_counts()

# create the charts

flag_axs = bake_pies([normal_flags, attack_flags], ['normal','attack'])        
plt.show()

#Attack Type vs Service Pie Chart
# get a series with the count of each service for attack and normal traffic

normal_services = train.loc[train.attack_flag == 0].service.value_counts()
attack_services = train.loc[train.attack_flag == 1].service.value_counts()

# create the charts

service_axs = bake_pies([normal_services, attack_services], ['normal','attack'])        
plt.show()
