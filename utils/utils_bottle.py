# TODO : 5 folder cross validation to handle overfitting problem
def get_specified_samples_MVTECAD(c,dec_x, dec_y):

    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    if (c == 0):
      return  [xbeg[0:209],xbeg[209:229]], [ybeg[0:209],ybeg[209:229]]
    else:
      return  [xbeg[0:22],xbeg[22:42]], [ybeg[0:22],ybeg[22:42]]

def cross_validation_5folds_MVTECAD(dec_x, dec_y):
  X_list = [[],[]]
  y_list = [[],[]]
  foldsX = []
  foldsY = []
  for i in range(0,2):
    X,y = get_specified_samples_MVTECAD(i,dec_x, dec_y)
    print(torch.bincount(y[0]))
    for j in range(0,2):
      X_list[j].append(X[j])
      y_list[j].append(y[j])

  for j in range(0,2):
    X = torch.cat(X_list[j])
    y = torch.cat(y_list[j])
    foldsX.append(X)
    foldsY.append(y)

  return foldsX, foldsY