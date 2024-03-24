def train_AE(dataset_torch, fold_id, data_name):
  torch.autograd.set_detect_anomaly(True)
  encoder = Encoder(args)
  decoder = Decoder(args)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(device)

  decoder = decoder.to(device)
  encoder = encoder.to(device)

  # decoder loss function
  criterion = nn.MSELoss()
  criterion = criterion.to(device)

  dl_batch_size = batch_size=dataset_torch.__len__()
  batch_size = args['batch_size']
  num_workers = 0
  train_loader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size,shuffle=True,num_workers=num_workers)
  dl_aux = torch.utils.data.DataLoader(dataset_torch, batch_size=dl_batch_size,shuffle=True,num_workers=num_workers)
  dec_x, dec_y = next(iter(dl_aux))
  del dl_aux

  best_loss = np.inf

  t0 = time.time()

  if args['train']:
      enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])       # optimization : Adam + learning rate = 0.0001
      dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])

      # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
      # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=30, gamma=0.5)

      for epoch in range(args['epochs']):                                       # number of training : 100
          train_loss = 0.0

          encoder.train()                                                       # train for one epoch -- set nets to train mode
          decoder.train()

          for images,labs in tqdm(train_loader):

            encoder.zero_grad()                                                 # zero gradients for each batch
            decoder.zero_grad()

            images = images.to(device)

            batch_size = images.size()[0]

            z_hat, mu, logvar = encoder(images)
            z_hat = z_hat.to(device)
            x_hat = decoder(z_hat)

            MSE = criterion(x_hat, images)

            # similar to F.1 example, Algorithm 2 outlined in appendix
            # of Kingma & Welling, 2014
            KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Penalty term
            tc = np.random.randint(10)                                        # class [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            xbeg = dec_x[dec_y == tc]                                         # image : only one class
            ybeg = dec_y[dec_y == tc]                                         # label : following above image

            xlen = len(xbeg)                                                  # 4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80
            nsamp = min(xlen, 100)                                            # 100 , 100 , 100 , 100, 100, 100, 100, 100, 60, 40 (number of sample)

            ind = np.random.choice(list(range(len(xbeg))),nsamp,replace=False)# index (100 indexes)
            xclass = xbeg[ind]                                                # tuple (images)
            yclass = ybeg[ind]                                                # tuple (labels)

            xclen = len(xclass)                                               # length of xclass (100/60/40)
            xcminus = np.arange(1,xclen)                                      # arrange like : [1, 2, ..., 100]
            xcplus = np.append(xcminus,0)                                     # append 0 into above list [..., 0]

            xcnew = np.array(xclass)                                          # Fix warning : converting list of numpy array to single numpy array create a single numpy array from the list of numpy arrays
            xcnew = xcnew[[xcplus], :]                                        # apply indexing
            xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
            xcnew = torch.Tensor(xcnew)
            xcnew = xcnew.to(device)

            xclass = torch.Tensor(xclass)                                     # encode xclass to feature space
            xclass = xclass.to(device)
            xclass, mu_2, logvar_2 = encoder(xclass)
            xclass = xclass.detach().cpu().numpy()
            xc_enc = (xclass[[xcplus],:])
            xc_enc = np.squeeze(xc_enc)
            xc_enc = torch.Tensor(xc_enc)
            xc_enc = xc_enc.to(device)

            ximg = decoder(xc_enc)                                            # decode xc_enc
            mse2 = criterion(ximg,xcnew)
            KLD2 = 0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())


            ELBO = MSE - KLD + mse2 - KLD2
            ELBO.backward()

            enc_optim.step()
            dec_optim.step()

            train_loss += ELBO.item()

          train_loss = train_loss/len(train_loader)

          print('Epoch: {} \tTrain Loss: {:.6f}'.format(epoch, train_loss))

          if train_loss < best_loss:
              # TODO : make directory if not exist
              path = '/content/drive/MyDrive/Colab/' + data_name + '/model/' + str(fold_id)
              if not os.path.exists(path):
                os.makedirs(path)

              path_enc = path + '/bst_enc.pth'
              path_dec = path + '/bst_dec.pth'
              torch.save(encoder.state_dict(), path_enc)
              torch.save(decoder.state_dict(), path_dec)
              best_loss = train_loss

  del dec_x
  del dec_y

  torch.cuda.empty_cache()

  t1 = time.time()
  print('total time(min): {:.2f}'.format((t1 - t0)/60))