import pandas as pd
import numpy as np

# Replace the paths of the train and test data with the local path
trainPath = r'D:\Study\Applied Machine Learning\Project\Datasets-20191109\train_imperson_without4n7_balanced_data.csv'
testPath = r'D:\Study\Applied Machine Learning\Project\Datasets-20191109\test_imperson_without4n7_balanced_data.csv'


def Preprocessing(path):
    origSet = pd.read_csv(path)
    
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
    
    return finalSet

def FeatureSelection(data):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    # Define our features and our target classifier
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]

    # First, remove all features with zero variance i.e. features with constant values. 
    X = X.loc[:,X.apply(pd.Series.nunique) != 1]

    # RFE with logistic regression
    model = LogisticRegression()

    # Try 10 features
    rfe_log = RFE(model, 10)
    fit_rfe_log = rfe_log.fit(X, Y)
    # Create list with names of features
    rfe_log_features = X.columns[fit_rfe_log.get_support()]

    # Print names of selected features
    for f in rfe_log_features:
        print(f)

    # Create csv file with only the relevant features
    cols_to_use = rfe_log_features.union(['class'])
    fs_output = data[cols_to_use]
#    fs_output.to_csv(r'FeatureSelectionOutput.csv', index=False)
#    
#    # Create csv file with test data and relevant features
#    fs_test = test[cols_to_use]
#    fs_test.to_csv(r'FeatureSelectionTestOutput.csv', index=False)
    
    return fs_output

def ModelSelection(trainData, testData):
    data_train = trainData
    data_train.shape
    
    data_train.head()
    
    # split values into inputs and outputs.
    values_train = data_train.values
    X_train = values_train[:,1:11]
    y_train = values_train[:,0]
    
    data_train.shape
    
    # Test data.
    data_test = testData
    
    # split values into inpits and outputs.
    values_test = data_test.values
    X_test = values_test[:,1:11]
    y_test = values_test[:,0]
    
    data_test.shape
    
    # RANDOM FOREST IS THE BEST PERFORMING ALGORYTHM
    
    ## RF with default hyperparameters
    
    # Initiate a RF model using default hyperparameters.
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    
    # Train model on train data.
    rf.fit(X_train, y_train)
    
    # Check model accuracy on the TEST set.
    rf_score = rf.score(X_test, y_test)
    print(rf_score)
    
    # Build confusion matrix.
    from sklearn.metrics import confusion_matrix
    rf_cm = confusion_matrix(y_test, rf.predict(X_test))
    print(rf_cm)
    
    ## RF hyperparameters tuning (Random Search)
    
    # Define a grid of hyperparameters.
    rf_params = { 'n_estimators': [1, 5, 10, 30, 50, 100, 200, 500], 
                 'max_depth': [None, 1, 2, 4, 8, 20, 50, 100], 
                 'min_samples_leaf': [1, 5, 10, 50, 100], 
                 'max_features': [None, 'auto', 'log2']
                }
    
    # Run random search.
    from sklearn.model_selection import RandomizedSearchCV
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, n_iter=25, 
                                   cv = 3, n_jobs=-1, random_state = 2019)
    
    import time
    start_time = time.time()
    rf_random.fit(X_train, y_train)
    finish_time = time.time()
    
    # Summarize results
    print("Best: %f using %s" % (rf_random.best_score_, rf_random.best_params_))
    print("Execution time: " + str((finish_time - start_time)))
    
    # Apply best values of hyperparameters to the model.
    rf_tuned = rf_random.best_estimator_
    
    # Train the tuned model on TRAIN set and check the accuracy
    rf_tuned.fit(X_train, y_train)
    rf_tuned_score = rf_tuned.score(X_test,y_test)
    print(rf_tuned_score)
    
    # Build confusion matrix.
    rf_tuned_cm = confusion_matrix(y_test, rf_tuned.predict(X_test))
    print(rf_tuned_cm)
    
    ## RF tuning Results
    
    print("RF default hyperparameters test accuracy: ", rf_score,', parameters: ', '\n', rf.get_params())
    print('Confusion matrix: ', '\n', rf_cm)
    print()
    print("RF tuned hyperparameters test accuracy: ", rf_tuned_score,', parameters: ', '\n', rf_tuned.get_params())
    print('Confusion matrix: ', '\n', rf_tuned_cm)

# Calling the functions sequentially
    
# Preprocess train and test sets
trainData = Preprocessing(trainPath)
testData = Preprocessing(testPath)

# Feature select train and test sets
FSTrainData = FeatureSelection(trainData)
FSTestData = FeatureSelection(testData)

# Show results
ModelSelection(FSTrainData, FSTestData)



