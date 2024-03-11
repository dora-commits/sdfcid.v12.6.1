# TODO : 5 folder cross validation to handle overfitting problem
def get_specified_samples(c,dec_x, dec_y):

    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    return  [xbeg[0:1200],xbeg[1200:2400],xbeg[2400:3600],xbeg[3600:4800],xbeg[4800:6000]], [ybeg[0:1200],ybeg[1200:2400],ybeg[2400:3600],ybeg[3600:4800],ybeg[4800:6000]]

def cross_validation_5folds(dec_x, dec_y):
  X_list = [[],[],[],[],[]]
  y_list = [[],[],[],[],[]]
  foldsX = []
  foldsY = []
  for i in range(0,10):
    X,y = get_specified_samples(i,dec_x, dec_y)
    print(torch.bincount(y[0]))
    for j in range(0,5):
      X_list[j].append(X[j])
      y_list[j].append(y[j])

  for j in range(0,5):
    X = torch.cat(X_list[j])
    y = torch.cat(y_list[j])
    foldsX.append(X)
    foldsY.append(y)

  return foldsX, foldsY

# TODO : create imbalanced dataset for training and generate samples
def make_imbalanced_dataset(X, y):
  imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
  new_X =[]
  new_Y =[]
  for c in range(0,10):
    xclass = X[y==c]
    yclass = y[y==c]
    new_X.append(xclass[0:imbal[c]])
    new_Y.append(yclass[0:imbal[c]])
  X_imbal = torch.cat(new_X)
  Y_imbal = torch.cat(new_Y)
  del new_X
  del new_Y
  return X_imbal, Y_imbal

def biased_get_class(c, dec_x, dec_y):

    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    return xbeg, ybeg
    #return xclass, yclass

# TODO : SMOTE Algorithm
def SMOTE(X, y,n_to_sample,cl):
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample