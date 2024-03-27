# TODO : Arguments
args = {}
args['dim_h']                 = 64                    
args['n_channel']             = 3                     
args['n_z']                   = 600                   # number of dimensions in latent space.
args['lr']                    = 0.0002                # learning rate for Adam optimizer .0002
args['epochs']                = 50                   
args['batch_size']            = 100                   # batch size for SGD, Adam
args['save']                  = True                  # save weights at each epoch of training if True
args['train']                 = True                  # train networks if True, else load networks from
args['resnet_learning_rate']  = 0.001                 # learning rate for Adam optimizer .001 with resnet18
args['resnet_num_epochs']     = 20                    
args['oversampling_method']   = 'ADASYN'              # Oversampling Algorithm
args['data_name']             = 'CIFAR10'             # Dataset
args['header']                = True

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