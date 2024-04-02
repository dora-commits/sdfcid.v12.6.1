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
  imbal = [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80]
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

# TODO : ADASYN Algorithm
def ADASYN(X, y,xclass, yclass,  cl, m_major, m_minor, beta=1):
  K = 5
  d= m_major/m_minor
  G = (m_major-m_minor)*beta

  # fitting the model
  clf = neighbors.KNeighborsClassifier()
  clf.fit(X, y)
  Ri = []
  Minority_per_xi = []

  for i in range(m_minor):
    # Returns indices of the closest neighbours, and return it as a list
    xi = xclass[i, :].reshape(1, -1)
    # Returns indices of the closest neighbours, and return it as a list
    neighbours = clf.kneighbors(xi, n_neighbors=K, return_distance=False)[0]
    delta=0
    for j in neighbours:
      if(y[j]!=0):
        delta+=1
    Ri.append(delta/K)

    minority = []
    for index in neighbours:
            # Shifted back 1 because indices start at 0
            if y[index]==cl:
                minority.append(index)
    Minority_per_xi.append(minority)

  Ri_norm = []
  for ri in Ri:
    ri_norm = ri / sum(Ri)
    Ri_norm.append(ri_norm)

  Gi = []
  for r in Ri_norm:
    gi = round(r * G)
    Gi.append(int(gi))
  syn_data=[]
  syn_number =0

  for i in range(m_minor):
    neighbor_indices = np.random.choice(list(range(1, K+1)),Gi[i])
    for j in range(Gi[i]):
        # If the minority list is not empty
        if Minority_per_xi[i]:
            index = np.random.choice(Minority_per_xi[i])
            xzi = X[index, :].reshape(1, -1)
            si = xi + (xzi - xi) * np.random.uniform(0, 1)
            syn_data.append(si)
            syn_number+=1
  return syn_data, [cl]*syn_number