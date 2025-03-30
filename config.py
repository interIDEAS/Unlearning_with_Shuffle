import os 


# Dataset ID

DATASET_ID = {
    ## Categorical classification datasets
    # 'ELECTRICITY': 44156, # 38474samples and 8features
    # 'EYE_MOVEMENTS': 44157, # 7608samples and 23features
    # 'KDD_UPSELLING': 44158, # 5032samples and 45features
    # 'COVERTYPE': 44159, # 423680samples and 54features
    # 'RL': 44160, # 4970samples and 12features 
    # 'ROAD_SAFETY': 44161, # 111762samples and 32features
    # 'COMPASS_ID': 44162, # 16644samples and 17features
    ## Numerical classification datasets
    # 'POL': 44122, # 10082samples and 26features 
    # 'HOUSE_16H': 44123, # 13488samples and 16features 
    # 'KDD_IPUMS': 44124, # 5188samples and 20features
    'MAGIC_TELE': 44125, # 13376samplesand 10features 
    # 'BANK_MARKETING': 44126, # 10578samples and 7features
    # 'PHONEME': 44127, # 3172samples and 5features 
    # 'MINIBOONE': 44128, # 72998samples and 50features
    # 'HIGGS': 44129, # 940160samples and 24features
    # 'JANNIS': 44131, # 57580samples and 54features
    'CREDIT': 44089, # 16714samples and 10features
    'CALI': 44090, # 20634samples and 8features 
    # 'WINE': 44091 # 2554samples and 11features
}

UNL_FEA_IDX = {
    # 'ELECTRICITY': [0, 1, 5, 6],
    # 'RL': [0, 1, 3, 4],
    # 'WINE': [0, 1, 9, 10],
    'CALI': [0, 1, 6, 7],
    'CREDIT': [0, 1, 8, 9],
    # 'PHONEME': [0, 1, 3, 4],
    # 'BANK_MARKETING': [0, 1, 4, 5],
    'MAGIC_TELE': [0, 1, 4, 5]
}



NUMERICAL_COLS = {
    'ELECTRICITY': ['nswprice', 'date', 'nswdemand', 'period', 'vicprice', 'transfer', 'vicdemand'], # len=7
    'RL': ['V21', 'V5', 'V6', 'V1', 'V20'], # len=5
    'WINE': ['alcohol', 'volatile acidity', 'free sulfur dioxide', 'total sulfur dioxide', 'chlorides', 'sulphates', 'residual sugar', 'fixed acidity', 'pH', 'citric acid',  'density'], # len=11
    'CALI': ['Latitude', 'Longitude', 'MedInc', 'AveOccup', 'AveRooms', 'HouseAge', 'AveBedrms', 'Population'], # len=8
    'CREDIT': ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'age', 'RevolvingUtilizationOfUnsecuredLines', 'NumberRealEstateLoansOrLines', 'MonthlyIncome', 'NumberOfDependents', 'DebtRatio'], # len=10
    'PHONEME': ['V1', 'V2', 'V3', 'V4', 'V5'], # len=5
    'BANK_MARKETING': ['V12', 'V14', 'V1', 'V6', 'V10', 'V13'], # len=6
    'MAGIC_TELE': ['fWidth:', 'fSize:', 'fConc:', 'fConc1:', 'fLength:', 'fAsym:'] # len=6
}


NOMINIAL_COLS = {
    'ELECTRICITY': ['day'], # len=1
    'RL': ['V8', 'V14', 'V15', 'V17', 'V18', 'V19'], # len=6
    'WINE': [],
    'CALI': [],
    'CREDIT': [],
    'PHONEME': [],
    'BANK_MARKETING': [],
    'MAGIC_TELE': []
}


OLD_NUMERICAL_COLS = {
    'ELECTRICITY': ['nswprice', 'period', 'transfer', 'date', 'nswdemand', 'vicprice', 'vicdemand'],
    'RL': ['V21', 'V1', 'V5', 'V6', 'V20'],
    'WINE': ['alcohol', 'free sulfur dioxide', 'pH', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'density', 'sulphates'],
    'CALI': ['Latitude', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Longitude'],
    'CREDIT': ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'age', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'],
    'PHONEME': ['V1', 'V2', 'V3', 'V4', 'V5'],
    'BANK_MARKETING': ['V12', 'V1', 'V6', 'V10', 'V13', 'V14'],
    'MAGIC_TELE': ['fWidth:', 'fLength:', 'fSize:', 'fConc:', 'fConc1:', 'fAsym:']
}


OLD_NOMINIAL_COLS = {
    'ELECTRICITY': ['day'],
    'RL': ['V8', 'V14', 'V15', 'V17', 'V18', 'V19'],
    'WINE': [],
    'CALI': [],
    'CREDIT': [],
    'PHONEME': [],
    'BANK_MARKETING': [],
    'MAGIC_TELE': []
}



MODEL_SAVE_FOLDER = 'model'
ACC_SAVE_FOLDER = 'results/acc'
TRAIN_LOSS_FOLDER = 'results/train_loss'
Y_PRED_FOLDER = 'results/y_pred'
RUNNING_TIME_FOLDER = 'results/running_time'

def get_ori_model_save_path(dataset_name: str, ori_training_epochs: int, model_type: str):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_epochs{}_ori_model{}.pth'.format(dataset_name, ori_training_epochs, model_type))

def get_ori_model_acc_save_path(dataset_name: str, ori_training_epochs: int, model_type: str):
    os.makedirs(ACC_SAVE_FOLDER, exist_ok=True)
    return os.path.join(ACC_SAVE_FOLDER, '{}_epochs{}_ori_model{}_acc.txt'.format(dataset_name, ori_training_epochs, model_type))

def get_ori_model_y_pred_save_path(dataset_name: str, ori_training_epochs: int, model_type: str):
    os.makedirs(Y_PRED_FOLDER, exist_ok=True)
    return os.path.join(Y_PRED_FOLDER, '{}_epochs{}_ori_model{}_y_pred.txt'.format(dataset_name, ori_training_epochs, model_type))

def get_ori_model_train_loss_save_path(dataset_name: str, ori_training_epochs: int, model_type: str):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_epochs{}_ori_model{}_train_loss.txt'.format(dataset_name, ori_training_epochs, model_type))

def get_ori_model_running_time_save_path(dataset_name: str, ori_training_epochs: int, model_type: str):
    os.makedirs(RUNNING_TIME_FOLDER, exist_ok=True)
    return os.path.join(RUNNING_TIME_FOLDER, '{}_epochs{}_ori_model{}_running_time.txt'.format(dataset_name, ori_training_epochs, model_type))


def get_RT_model_save_path(dataset_name: str, RT_training_epochs: int, RT_train_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_epochs{}_traintimesidx{}_RT_model{}_unlfeaidx{}.pth'.format(dataset_name, RT_training_epochs, RT_train_times_idx, model_type, unl_fea_idx))

def get_RT_model_acc_save_path(dataset_name: str, RT_training_epochs: int, model_type: str, unl_fea_idx: int):
    os.makedirs(ACC_SAVE_FOLDER, exist_ok=True)
    return os.path.join(ACC_SAVE_FOLDER, '{}_epochs{}_RT_model{}_unlfeaidx{}_acc.txt'.format(dataset_name, RT_training_epochs, model_type, unl_fea_idx))

def get_RT_model_y_pred_save_path(dataset_name: str, RT_training_epochs: int, RT_train_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(Y_PRED_FOLDER, exist_ok=True)
    return os.path.join(Y_PRED_FOLDER, '{}_epochs{}_traintimesidx{}_RT_model{}_unlfeaidx{}_y_pred.txt'.format(dataset_name, RT_training_epochs, RT_train_times_idx, model_type, unl_fea_idx))

def get_RT_model_train_loss_save_path(dataset_name: str, RT_training_epochs: int, RT_train_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_epochs{}_traintimesidx{}_RT_model{}_unlfeaidx{}_train_loss.txt'.format(dataset_name, RT_training_epochs, RT_train_times_idx, model_type, unl_fea_idx))

def get_RT_model_running_time_save_path(dataset_name: str, RT_training_epochs: int, model_type: str, unl_fea_idx: int):
    os.makedirs(RUNNING_TIME_FOLDER, exist_ok=True)
    return os.path.join(RUNNING_TIME_FOLDER, '{}_epochs{}_RT_model{}_unlfeaidx{}_running_time.txt'.format(dataset_name, RT_training_epochs, model_type, unl_fea_idx))



def get_ori_BL3RepDetExtractormodel_save_path(dataset_name: str, ori_BL3_training_epochs: int, model_type: str):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_epochs{}_ori_BL3RepDetExtractor_model{}.pth'.format(dataset_name, ori_BL3_training_epochs, model_type))

def get_ori_BL3Classifier_model_save_path(dataset_name: str, ori_BL3_training_epochs: int, model_type: str):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_epochs{}_ori_BL3Classifier_model{}.pth'.format(dataset_name, ori_BL3_training_epochs, model_type))
def get_ori_BL3_model_train_loss_save_path(dataset_name: str, ori_BL3_training_epochs: int, model_type: str):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_epochs{}_ori_BL3_model{}_train_loss.txt'.format(dataset_name, ori_BL3_training_epochs, model_type))
    


def get_BL1_model_save_path(dataset_name: str, ori_training_epochs: int, BL1_unlearning_epochs: int, unlearn_times_idx: int):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_BL1.pth'.format(dataset_name, ori_training_epochs, BL1_unlearning_epochs, unlearn_times_idx))

def get_BL1_model_acc_save_path(dataset_name: str, ori_training_epochs: int, BL1_unlearning_epochs: int):
    os.makedirs(ACC_SAVE_FOLDER, exist_ok=True)
    return os.path.join(ACC_SAVE_FOLDER, '{}_oriep{}_ulep{}_BL1_acc.txt'.format(dataset_name, ori_training_epochs, BL1_unlearning_epochs))

def get_BL1_model_y_pred_save_path(dataset_name: str, ori_training_epochs: int, BL1_unlearning_epochs: int, unlearn_times_idx: int):
    os.makedirs(Y_PRED_FOLDER, exist_ok=True)
    return os.path.join(Y_PRED_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_BL1_y_pred.txt'.format(dataset_name, ori_training_epochs, BL1_unlearning_epochs, unlearn_times_idx))

def get_BL1_model_train_loss_save_path(dataset_name: str, ori_training_epochs: int, BL1_unlearning_epochs: int, unlearn_times_idx: int):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_BL1_train_loss.txt'.format(dataset_name, ori_training_epochs, BL1_unlearning_epochs, unlearn_times_idx))

def get_BL1_model_running_time_save_path(dataset_name: str, ori_training_epochs: int, BL1_unlearning_epochs: int):
    os.makedirs(RUNNING_TIME_FOLDER, exist_ok=True)
    return os.path.join(RUNNING_TIME_FOLDER, '{}_oriep{}_ulep{}_BL1_running_time.txt'.format(dataset_name, ori_training_epochs, BL1_unlearning_epochs))


def get_BL2_model_save_path(dataset_name: str, ori_training_epochs: int, BL2_unlearning_epochs: int, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_BL2_model{}_unlfeaidx{}.pth'.format(dataset_name, ori_training_epochs, BL2_unlearning_epochs, unlearn_times_idx, model_type, unl_fea_idx))

def get_BL2_model_acc_save_path(dataset_name: str, ori_training_epochs: int, BL2_unlearning_epochs: int, model_type: str, unl_fea_idx: int):
    os.makedirs(ACC_SAVE_FOLDER, exist_ok=True)
    return os.path.join(ACC_SAVE_FOLDER, '{}_oriep{}_ulep{}_BL2_model{}_unlfeaidx{}_acc.txt'.format(dataset_name, ori_training_epochs, BL2_unlearning_epochs, model_type, unl_fea_idx))

def get_BL2_model_y_pred_save_path(dataset_name: str, ori_training_epochs: int, BL2_unlearning_epochs: int, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(Y_PRED_FOLDER, exist_ok=True)
    return os.path.join(Y_PRED_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_BL2_model{}_unlfeaidx{}_y_pred.txt'.format(dataset_name, ori_training_epochs, BL2_unlearning_epochs, unlearn_times_idx, model_type, unl_fea_idx))

def get_BL2_model_train_loss_save_path(dataset_name: str, ori_training_epochs: int, BL2_unlearning_epochs: int, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_BL2_model{}_unlfeaidx{}_train_loss.txt'.format(dataset_name, ori_training_epochs, BL2_unlearning_epochs, unlearn_times_idx, model_type, unl_fea_idx))

def get_BL2_model_running_time_save_path(dataset_name: str, ori_training_epochs: int, BL2_unlearning_epochs: int, model_type: str, unl_fea_idx: int):
    os.makedirs(RUNNING_TIME_FOLDER, exist_ok=True)
    return os.path.join(RUNNING_TIME_FOLDER, '{}_oriep{}_ulep{}_BL2_model{}_unlfeaidx{}_running_time.txt'.format(dataset_name, ori_training_epochs, BL2_unlearning_epochs, model_type, unl_fea_idx))



def get_BL3DecoderMIhx_model_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, model_type: str):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_MIhxep{}_BL3DecoderMIhx_model{}.pth'.format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, model_type))

def get_BL3MIhy_model_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3MIhy_training_epochs: int, model_type: str):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_MIhyep{}_BL3MIhy_model{}.pth'.format(dataset_name, ori_BL3_training_epochs, BL3MIhy_training_epochs, model_type))

def get_BL3MIhz_model_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3MIhz_training_epochs: int, model_type: str):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_MIhzep{}_BL3MIhz_model{}.pth'.format(dataset_name, ori_BL3_training_epochs, BL3MIhz_training_epochs, model_type))

def get_BL3_prep_running_time_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, model_type: str):
    os.makedirs(RUNNING_TIME_FOLDER, exist_ok=True)
    return os.path.join(RUNNING_TIME_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_BL3_model{}_prep_running_time.txt'.format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, model_type))



def get_BL3RepDetExtractor_model_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, BL3_unlearning_epochs: int, BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_ulep{}_l1{}_l2{}_l3{}_ultimesidx{}_BL3RepDetExtractor_model{}_unlfeaidx{}.pth'\
        .format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, BL3_unlearning_epochs, BL3_lamda1, BL3_lamda2, BL3_lamda3, unlearn_times_idx, model_type, unl_fea_idx))
    
def get_BL3Classifier_model_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, BL3_unlearning_epochs: int, BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_ulep{}_l1{}_l2{}_l3{}_ultimesidx{}_BL3Classifier_model{}_unlfeaidx{}.pth'\
        .format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, BL3_unlearning_epochs, BL3_lamda1, BL3_lamda2, BL3_lamda3, unlearn_times_idx, model_type, unl_fea_idx))
    
def get_BL3_model_acc_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, BL3_unlearning_epochs: int, BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float, model_type: str, unl_fea_idx: int):
    os.makedirs(ACC_SAVE_FOLDER, exist_ok=True)
    return os.path.join(ACC_SAVE_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_ulep{}_l1{}_l2{}_l3{}_BL3_model{}_unlfeaidx{}_acc.txt'\
        .format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, BL3_unlearning_epochs, BL3_lamda1, BL3_lamda2, BL3_lamda3, model_type, unl_fea_idx)) 

def get_BL3_model_y_pred_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, BL3_unlearning_epochs: int, BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(Y_PRED_FOLDER, exist_ok=True)
    return os.path.join(Y_PRED_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_ulep{}_l1{}_l2{}_l3{}_ultimesidx{}_BL3_model{}_unlfeaidx{}_y_pred.txt'\
        .format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, BL3_unlearning_epochs, BL3_lamda1, BL3_lamda2, BL3_lamda3, unlearn_times_idx, model_type, unl_fea_idx)) 

def get_BL3_model_train_task_loss_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, BL3_unlearning_epochs: int, BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_ulep{}_l1{}_l2{}_l3{}_ultimesidx{}_BL3_model{}_unlfeaidx{}_train_task_loss.txt'\
        .format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, BL3_unlearning_epochs, BL3_lamda1, BL3_lamda2, BL3_lamda3, unlearn_times_idx, model_type, unl_fea_idx)) 

def get_BL3_model_train_MI_loss_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, BL3_unlearning_epochs: int, BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_ulep{}_l1{}_l2{}_l3{}_ultimesidx{}_BL3_model{}_unlfeaidx{}_train_MI_loss.txt'\
        .format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, BL3_unlearning_epochs, BL3_lamda1, BL3_lamda2, BL3_lamda3, unlearn_times_idx, model_type, unl_fea_idx)) 


def get_BL3_model_running_time_save_path(dataset_name: str, ori_BL3_training_epochs: int, BL3DecoderMIhx_training_epochs: int, BL3MIhy_training_epochs: int, BL3MIhz_training_epochs: int, BL3_unlearning_epochs: int, BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float, model_type: str, unl_fea_idx: int):
    os.makedirs(RUNNING_TIME_FOLDER, exist_ok=True)
    return os.path.join(RUNNING_TIME_FOLDER, '{}_oriep{}_MIhxep{}_MIhyep{}_MIhzep{}_ulep{}_l1{}_l2{}_l3{}_BL3_model{}_unlfeaidx{}_runnnig_time.txt'\
        .format(dataset_name, ori_BL3_training_epochs, BL3DecoderMIhx_training_epochs, BL3MIhy_training_epochs, BL3MIhz_training_epochs, BL3_unlearning_epochs, BL3_lamda1, BL3_lamda2, BL3_lamda3, model_type, unl_fea_idx))
    
    
def get_our_model_save_path(dataset_name: str, ori_training_epochs: int, unlearning_epochs: int, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    return os.path.join(MODEL_SAVE_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_our_model{}_unlfeaidx{}.pth'.format(dataset_name, ori_training_epochs, unlearning_epochs, unlearn_times_idx, model_type, unl_fea_idx))

def get_our_model_acc_save_path(dataset_name: str, ori_training_epochs: int, unlearning_epochs: int, model_type: str, unl_fea_idx: int):
    os.makedirs(ACC_SAVE_FOLDER, exist_ok=True)
    return os.path.join(ACC_SAVE_FOLDER, '{}_oriep{}_ulep{}_our_model{}_unlfeaidx{}_acc.txt'.format(dataset_name, ori_training_epochs, unlearning_epochs, model_type, unl_fea_idx))

def get_our_model_y_pred_save_path(dataset_name: str, ori_training_epochs: int, unlearning_epochs: int, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(Y_PRED_FOLDER, exist_ok=True)
    return os.path.join(Y_PRED_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_our_model{}_unlfeaidx{}_y_pred.txt'.format(dataset_name, ori_training_epochs, unlearning_epochs, unlearn_times_idx, model_type, unl_fea_idx)) 

def get_our_model_train_loss_save_path(dataset_name: str, ori_training_epochs: int, unlearning_epochs: int, unlearn_times_idx: int, model_type: str, unl_fea_idx: int):
    os.makedirs(TRAIN_LOSS_FOLDER, exist_ok=True)
    return os.path.join(TRAIN_LOSS_FOLDER, '{}_oriep{}_ulep{}_ultimesidx{}_our_model{}_unlfeaidx{}_train_loss.txt'.format(dataset_name, ori_training_epochs, unlearning_epochs, unlearn_times_idx, model_type, unl_fea_idx))

def get_our_model_running_time_save_path(dataset_name: str, ori_training_epochs: int, unlearning_epochs: int, model_type: str, unl_fea_idx: int):
    os.makedirs(RUNNING_TIME_FOLDER, exist_ok=True)
    return os.path.join(RUNNING_TIME_FOLDER, '{}_oriep{}_ulep{}_our_model{}_unlfeaidx{}_running_time.txt'.format(dataset_name, ori_training_epochs, unlearning_epochs, model_type, unl_fea_idx)) 