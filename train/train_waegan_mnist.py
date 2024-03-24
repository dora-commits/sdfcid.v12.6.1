def train_AE(dataset_torch, fold_id, data_name):
  torch.autograd.set_detect_anomaly(True)
  encoder = Encoder(args)
  decoder = Decoder(args)
  discriminator = Discriminator(args)


  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(device)

  decoder = decoder.to(device)
  encoder = encoder.to(device)
  discriminator = discriminator.to(device)

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
      enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])       # optimization : Adam + learning rate = 0.0002
      dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])
      dis_optim = torch.optim.Adam(discriminator.parameters(), lr = args['lr'])

    #   enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
    #   dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=30, gamma=0.5)
    #   dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size=30, gamma=0.5)

      # one and -one allow us to control descending / ascending gradient descent
      one = torch.tensor(1, dtype=torch.float)
      one = one.to(device)

      for epoch in range(args['epochs']):                                       # number of training : 100
          train_loss = 0.0

          encoder.train()                                                       # train for one epoch -- set nets to train mode
          decoder.train()
          discriminator.train()


          for images,labs in tqdm(train_loader):

            encoder.zero_grad()                                               # zero gradients for each batch
            decoder.zero_grad()
            discriminator.zero_grad()

            images = images.to(device)

            #### TRAIN DISCRIMINATOR ####

            # freeze auto encoder params
            frozen_params(decoder)
            frozen_params(encoder)

            # free discriminator params
            free_params(discriminator)

            # run discriminator against randn draws
            z = torch.randn(images.size()[0], args['n_z']) * args['sigma']
            d_z = discriminator(z.to(device))

            # run discriminator against encoder z's
            z_hat = encoder(images)
            d_z_hat = discriminator(z_hat)

            d_z_loss = args['lambda']*torch.log(d_z).mean()

            d_z_hat_clam = torch.clamp(d_z_hat, min=1e-7, max=1-1e-7)  # clamp outputs to range [1e-6, 1-1e-6]
            d_z_hat_loss = args['lambda']*torch.log(1-d_z_hat_clam).mean()

            d_z_loss.backward(-one)
            d_z_hat_loss.backward(-one)

            dis_optim.step()

            #### TRAIN GENERATOR ####

            # flip which networks are frozen, which are not
            free_params(decoder)
            free_params(encoder)
            frozen_params(discriminator)

            batch_size = images.size()[0]

            # run images
            z_hat = encoder(images)
            x_hat = decoder(z_hat)

            # discriminate latents
            z_hat2 = encoder(Variable(images.data))
            d_z_hat = discriminator(z_hat2)

            MSE = criterion(x_hat, images)

            d_z_hat_clamped = torch.clamp(d_z_hat, min=1e-7, max=1-1e-7)  # clamp outputs to range [1e-6, 1-1e-6]
            d_loss = args['lambda'] * (torch.log(d_z_hat_clamped)).mean()

            comb_loss = MSE - d_loss

            MSE.backward(one)

            d_loss.backward(-one)

            enc_optim.step()
            dec_optim.step()

            train_loss += comb_loss.item()

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