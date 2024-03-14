if (args['resnet_learning_rate'] == 0.001):
  figure, axis = plt.subplots(1, 5, figsize=(15,3))
  x = []
  for i in range(5):
    x.append([str(m) for m in range(0, len(train_loss_total[i]))])
  for i in range(1):
    for j in range(5):
      axis[j].plot(x[j], train_loss_total[j]       , label = "train loss")
      axis[j].plot(x[j], validation_loss_total[j]  , label = "validation loss")
      if (j == 0):
        axis[j].legend()
      axis[j].set_title("Folder {}".format(j))
  plt.tight_layout()
  plt.show()
  
if (args['resnet_learning_rate'] == 0.001):
  figure, axis = plt.subplots(1, 5, figsize=(15,3))
  x = []
  for i in range(5):
    x.append([str(m) for m in range(0, len(train_loss_total[i]))])
  for i in range(1):
    for j in range(5):
      axis[j].plot(x[j], train_acc_total[j]       , label = "train acc")
      axis[j].plot(x[j], validation_acc_total[j]  , label = "validation acc")
      if (j == 0):
        axis[j].legend()
      axis[j].set_title("Folder {}".format(j))
  plt.tight_layout()
  plt.show()