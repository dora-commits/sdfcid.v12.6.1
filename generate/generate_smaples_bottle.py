def GenerateSamples(dataset_torch, fold_id, data_name):
  np.printoptions(precision=5,suppress=True)

  modpth = '/content/drive/MyDrive/Colab/' + data_name + '/model/'
  encf = []
  decf = []
  for p in range(1):
      enc = modpth + str(p) + '/bst_enc_SMOTEAE_Bottle_1206.pth'
      dec = modpth + str(p) + '/bst_dec_SMOTEAE_Bottle_1206.pth'
      encf.append(enc)
      decf.append(dec)

  dl_batch_size = batch_size=dataset_torch.__len__()
  batch_size = args['batch_size']
  num_workers = 0
  dl_aux = torch.utils.data.DataLoader(dataset_torch, batch_size=dl_batch_size,shuffle=True,num_workers=num_workers)
  dec_x, dec_y = next(iter(dl_aux))
  del dl_aux

  classes = ('0', '1')

  train_on_gpu = torch.cuda.is_available()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  path_enc = encf[fold_id]
  path_dec = decf[fold_id]

  encoder = Encoder(args)
  encoder.load_state_dict(torch.load(path_enc), strict=False)
  encoder = encoder.to(device)

  decoder = Decoder(args)
  decoder.load_state_dict(torch.load(path_dec), strict=False)
  decoder = decoder.to(device)

  encoder.eval()
  decoder.eval()

  imbal = [209, 22]
  resx = []
  resy = []

  xclasses=[]
  yclasses=[]

  for i in range(0,2):
    xclass, yclass = biased_get_class(i, dec_x, dec_y)
    xclass = torch.Tensor(xclass)
    xclass = xclass.to(device)
    xclass = encoder(xclass)
    xclass = xclass.detach().cpu().numpy()
    xclasses.append(xclass)
    yclasses.append(yclass)
  allX = np.concatenate(xclasses)
  allY = np.concatenate(yclasses)

  for i in range(1,2):
      xclass, yclass = xclasses[i], yclasses[i]
      print('Class {}'.format(i))
      print("Len Class", len(yclass))
      n = imbal[0] - imbal[i]
      if(args['oversampling_method'] == 'SMOTE'):
        xsamp, ysamp = SMOTE(xclass,yclass,n,i)
      elif(args['oversampling_method'] == 'ADASYN'):
        xsamp, ysamp = ADASYN(allX, allY, xclass, yclass,  i, imbal[0], imbal[i], beta=1)

      ysamp = np.array(ysamp)
      xsamp = torch.Tensor(xsamp)
      xsamp = xsamp.to(device)
      ximg = decoder(xsamp)

      ximn = ximg.detach().cpu().numpy()                                        # (4000, 3, 32, 32)

      resx.append(ximn)
      resy.append(ysamp)

      del xclass
      del yclass
      del xsamp
      del ysamp
      del ximn

  del xclasses
  del yclasses
  del allX
  del allY

  resx1 = np.vstack(resx)
  resy1 = np.hstack(resy)
  del resx
  del resy

  resx1 = resx1.reshape(resx1.shape[0],-1)                                      # (..., 784)

  dec_x1 = dec_x.reshape(dec_x.shape[0],-1)

  combx = np.vstack((resx1,dec_x1))
  comby = np.hstack((resy1,dec_y))

  del resx1
  del resy1
  del dec_x1
  del dec_y
  del dec_x

  print(combx.shape)                                                            # (40000, 3, 32, 32)
  print(comby.shape)                                                            # (40000,)

  torch.cuda.empty_cache()
  return combx, comby