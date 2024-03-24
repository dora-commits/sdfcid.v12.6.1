# TODO : Arguments
args = {}
args['dim_h']                 = 64                    
args['n_channel']             = 1                     # number of channels in the input data
args['n_z']                   = 300                   # number of dimensions in latent space.
args['lr']                    = 0.0002                # learning rate for Adam optimizer .0002
args['epochs']                = 50                    
args['batch_size']            = 100                   
args['save']                  = True                  
args['train']                 = True                  
args['resnet_learning_rate']  = 0.001                 # learning rate for Adam optimizer .001 with resnet18
args['resnet_num_epochs']     = 20                    
args['oversampling_method']   = 'SMOTE'               # Oversampling Algorithm
args['data_name']             = 'MNIST'               # Dataset
args['header']                = True
args['sigma']                 = 1.0
args['lambda']                = 0.01

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