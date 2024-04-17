# TODO : Arguments
args = {}
args['dim_h']                 = 64                    
args['n_channel']             = 3                     # number of channels in the input data
args['n_z']                   = 800                   
args['lr']                    = 0.0002                # learning rate for Adam optimizer .0002
args['epochs']                = 100                   
args['batch_size']            = 21                    # batch size for SGD, Adam
args['save']                  = True                  # save weights at each epoch of training if True
args['train']                 = True                  # train networks if True, else load networks from
args['resnet_learning_rate']  = 0.001                 # learning rate for Adam optimizer .001 with resnet18
args['resnet_num_epochs']     = 20                    
args['oversampling_method']   = 'SMOTE'               # Oversampling Algorithm
args['data_name']             = 'MVTECAD'             # Dataset
args['header']                = True
args['sigma']                 = 1.0
args['lambda']                = 0.01
args['pre_trained']           = False

# TODO : Measuring Metrics
num_repeats = 1
Accuracy      = np.zeros(num_repeats)
GeometricMean = np.zeros(num_repeats)
F1Score       = np.zeros(num_repeats)
Precision     = np.zeros(num_repeats)
Recall        = np.zeros(num_repeats)

result_df = pd.DataFrame(columns=['Accuracy', 'GeometricMean', 'F1Score', 'Precision', 'Recall'])

# TODO : Check overfitting problem
train_loss_total = []
validation_loss_total = []
train_acc_total = []
validation_acc_total = []


# TODO : Time
train_model_time           = np.zeros(num_repeats)
generate_samples_time      = np.zeros(num_repeats)
train_classification_time  = np.zeros(num_repeats)
prediction_time            = np.zeros(num_repeats)

time_df = pd.DataFrame(columns=['Train model time', 'Generate samples time', 'Train classification time', 'Prediction time'])
     