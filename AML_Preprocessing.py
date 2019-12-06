import pandas as pd
import numpy as np

# Import data set as is
#url = r'https://raw.githubusercontent.com/korlov01/AML-Project/master/train_imperson_without4n7_balanced_data.csv?token=ANXOWMTGDZ27RJD6LKVQNSC54AIJG'
origSet = pd.read_csv(r'D:\Study\Applied Machine Learning\Project\Datasets-20191109\test_imperson_without4n7_balanced_data.csv')

# List of columns for the data set, except frame.time_epoch and frame.time_relative
colList = ['frame.interface_id','frame.dlt','frame.offset_shift','frame.time_delta','frame.time_delta_displayed','frame.len','frame.cap_len','frame.marked','frame.ignored','radiotap.version','radiotap.pad','radiotap.length','radiotap.present.tsft','radiotap.present.flags','radiotap.present.rate','radiotap.present.channel','radiotap.present.fhss','radiotap.present.dbm_antsignal','radiotap.present.dbm_antnoise','radiotap.present.lock_quality','radiotap.present.tx_attenuation','radiotap.present.db_tx_attenuation','radiotap.present.dbm_tx_power','radiotap.present.antenna','radiotap.present.db_antsignal','radiotap.present.db_antnoise','radiotap.present.rxflags','radiotap.present.xchannel','radiotap.present.mcs','radiotap.present.ampdu','radiotap.present.vht','radiotap.present.reserved','radiotap.present.rtap_ns','radiotap.present.vendor_ns','radiotap.present.ext','radiotap.mactime','radiotap.flags.cfp','radiotap.flags.preamble','radiotap.flags.wep','radiotap.flags.frag','radiotap.flags.fcs','radiotap.flags.datapad','radiotap.flags.badfcs','radiotap.flags.shortgi','radiotap.datarate','radiotap.channel.freq','radiotap.channel.type.turbo','radiotap.channel.type.cck','radiotap.channel.type.ofdm','radiotap.channel.type.2ghz','radiotap.channel.type.5ghz','radiotap.channel.type.passive','radiotap.channel.type.dynamic','radiotap.channel.type.gfsk','radiotap.channel.type.gsm','radiotap.channel.type.sturbo','radiotap.channel.type.half','radiotap.channel.type.quarter','radiotap.dbm_antsignal','radiotap.antenna','radiotap.rxflags.badplcp','wlan.fc.type_subtype','wlan.fc.version','wlan.fc.type','wlan.fc.subtype','wlan.fc.ds','wlan.fc.frag','wlan.fc.retry','wlan.fc.pwrmgt','wlan.fc.moredata','wlan.fc.protected','wlan.fc.order','wlan.duration','wlan.ra','wlan.da','wlan.ta','wlan.sa','wlan.bssid','wlan.frag','wlan.seq','wlan.bar.type','wlan.ba.control.ackpolicy','wlan.ba.control.multitid','wlan.ba.control.cbitmap','wlan.bar.compressed.tidinfo','wlan.ba.bm','wlan.fcs_good','wlan_mgt.fixed.capabilities.ess','wlan_mgt.fixed.capabilities.ibss','wlan_mgt.fixed.capabilities.cfpoll.ap','wlan_mgt.fixed.capabilities.privacy','wlan_mgt.fixed.capabilities.preamble','wlan_mgt.fixed.capabilities.pbcc','wlan_mgt.fixed.capabilities.agility','wlan_mgt.fixed.capabilities.spec_man','wlan_mgt.fixed.capabilities.short_slot_time','wlan_mgt.fixed.capabilities.apsd','wlan_mgt.fixed.capabilities.radio_measurement','wlan_mgt.fixed.capabilities.dsss_ofdm','wlan_mgt.fixed.capabilities.del_blk_ack','wlan_mgt.fixed.capabilities.imm_blk_ack','wlan_mgt.fixed.listen_ival','wlan_mgt.fixed.current_ap','wlan_mgt.fixed.status_code','wlan_mgt.fixed.timestamp','wlan_mgt.fixed.beacon','wlan_mgt.fixed.aid','wlan_mgt.fixed.reason_code','wlan_mgt.fixed.auth.alg','wlan_mgt.fixed.auth_seq','wlan_mgt.fixed.category_code','wlan_mgt.fixed.htact','wlan_mgt.fixed.chanwidth','wlan_mgt.fixed.fragment','wlan_mgt.fixed.sequence','wlan_mgt.tagged.all','wlan_mgt.ssid','wlan_mgt.ds.current_channel','wlan_mgt.tim.dtim_count','wlan_mgt.tim.dtim_period','wlan_mgt.tim.bmapctl.multicast','wlan_mgt.tim.bmapctl.offset','wlan_mgt.country_info.environment','wlan_mgt.rsn.version','wlan_mgt.rsn.gcs.type','wlan_mgt.rsn.pcs.count','wlan_mgt.rsn.akms.count','wlan_mgt.rsn.akms.type','wlan_mgt.rsn.capabilities.preauth','wlan_mgt.rsn.capabilities.no_pairwise','wlan_mgt.rsn.capabilities.ptksa_replay_counter','wlan_mgt.rsn.capabilities.gtksa_replay_counter','wlan_mgt.rsn.capabilities.mfpr','wlan_mgt.rsn.capabilities.mfpc','wlan_mgt.rsn.capabilities.peerkey','wlan_mgt.tcprep.trsmt_pow','wlan_mgt.tcprep.link_mrg','wlan.wep.iv','wlan.wep.key','wlan.wep.icv','wlan.tkip.extiv','wlan.ccmp.extiv','wlan.qos.tid','wlan.qos.priority','wlan.qos.eosp','wlan.qos.ack','wlan.qos.amsdupresent','wlan.qos.buf_state_indicated','wlan.qos.bit4','wlan.qos.txop_dur_req','wlan.qos.buf_state_indicated','data.len','class']

# Replace columns numbers with feature names and store in a new data set
origSetCols = origSet.columns.tolist()
nameDictionary = dict(zip(origSetCols, colList))
# New set with feature names
origSetWithColNames = origSet.rename(columns=nameDictionary)

# Identify columns with blank values
colsWithBlanks = origSetWithColNames.columns[origSetWithColNames.isnull().any()].tolist()

# Replace blanks, if any, with medians
if len(colsWithBlanks) > 0:
    origSetWithColNames.fillna(origSetWithColNames.median())
else:
    pass

# Change categorical features to numeric features, if any
colTypes = [str(w) for w in origSetWithColNames.dtypes.tolist()]

if 'object' in colTypes:
    objCols = origSetWithColNames.columns[origSetWithColNames.dtypes == object].tolist()
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    le = LabelEncoder()
    origSetWithColNames.loc[:, objCols] = le.fit_transform(origSetWithColNames.loc[:, objCols])
    ohe = OneHotEncoder(categorical_features=[origSetWithColNames.columns.get_loc(c) for c in objCols])
    origSetWithColNames = ohe.fit_transform(origSetWithColNames)
else:
    pass

# Standardize the data using RobustScaler
origSetWithColNamesWithoutClass = origSetWithColNames.iloc[:, :-1]
classSet = origSetWithColNames.iloc[:, -1]
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
finalSet = pd.DataFrame(scaler.fit_transform(origSetWithColNamesWithoutClass), columns=colList[:-1])
finalSet['class'] = classSet

finalSet.to_csv(r'D:\Study\Applied Machine Learning\Project\ScaledTestDataSet.csv', index=False)


