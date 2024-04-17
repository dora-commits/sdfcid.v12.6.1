if not args['pre_trained']:
  for repeats in range(0,num_repeats):

      acs_array_smote     = np.zeros(1)
      gm_array_smote      = np.zeros(1)
      f1macro_array_smote = np.zeros(1)
      precision_smote     = np.zeros(1)
      recall_smote        = np.zeros(1)


      for i in range(0, 1):
          aux_X = FoldsX.copy()
          aux_Y = FoldsY.copy()

          print('Fold {}'.format(i + 1))

          trainX_imbal = FoldsX[0]
          trainY_imbal = FoldsY[0]

          testX = FoldsX[1]
          testY = FoldsY[1]

          dataset_train = torch.utils.data.TensorDataset(trainX_imbal, trainY_imbal)

          t0 = time.time()
          #train_AE(dataset_train, i, args['data_name'])
          t1 = time.time()

          t2 = time.time()
          # combx, comby = GenerateSamples(dataset_train, i, args['data_name'])
          t3 = time.time()
          # combx_v = combx
          # comby_v = comby

          t4 = time.time()
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          model = ResNetClassifier()
          model.to(device)

          # combx = combx.reshape(combx.shape[0],3, 64, 64)
          # combx = torch.Tensor(combx)
          # comby = torch.tensor(comby,dtype=torch.long)

          x_train, x_validation, y_train, y_validation = train_test_split(trainX_imbal, trainY_imbal, test_size=0.2, random_state=0, stratify=trainY_imbal)

          train_set = TensorDataset(x_train, y_train)
          train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'],shuffle=True,num_workers=1)

          validation_set = TensorDataset(x_validation, y_validation)
          validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args['batch_size'],shuffle=True,num_workers=1)

          train_loss_step, train_acc_step, validation_loss_step, validation_acc_step = model.train_model(train_loader, validation_loader, args['resnet_num_epochs'], args['resnet_learning_rate'], args['data_name'], i)
          t5 = time.time()

          print('Total: '     , trainX_imbal.shape)
          print('Total class:'      , collections.Counter(trainY_imbal.numpy()))
          print('train set total: ', x_train.shape)
          print('train set class:' , collections.Counter(y_train.numpy()))
          print('validation set total: '      , x_validation.shape)
          print('validation set class:'       , collections.Counter(y_validation.numpy()))

          train_loss_total.append(train_loss_step)
          train_acc_total.append(train_acc_step)
          validation_loss_total.append(validation_loss_step)
          validation_acc_total.append(validation_acc_step)

          t6 = time.time()
          resnet_outputs = model.predict_model(testX.to(device))
          torch.cuda.empty_cache()

          y_predict = [torch.argmax(resnet_outputs[i]).item() for i in range(resnet_outputs.shape[0])]
          y_true = [testY[i].item() for i in range(resnet_outputs.shape[0])]
          t7 = time.time()

          train_model_time[repeats]           = np.round((t1 - t0)/60, 5)
          generate_samples_time[repeats]      = np.round((t3 - t2)/60, 5)
          train_classification_time[repeats]  = np.round((t5 - t4)/60, 5)
          prediction_time[repeats]            = np.round((t7 - t6)/60, 5)

          time_df.loc[len(time_df.index)] = [train_model_time[repeats],generate_samples_time[repeats],train_classification_time[repeats], prediction_time[repeats]]

          # fig, ax = plt.subplots()
          fig, ax = plt.subplots(figsize=(15, 15))
          fig.patch.set_visible(False)
          ax.axis('off')
          ax.axis('tight')
          ax.table(cellText=time_df.values, colLabels=time_df.columns, loc='center')
          fig.tight_layout()
          plt.show()

          acc = accuracy_score(y_true, y_predict)
          gm = geometric_mean_score(y_true, y_predict, average='macro')
          f1_macro = f1_score(y_true, y_predict, average='macro')
          precision = precision_score(y_true, y_predict, average='macro')
          recall = recall_score(y_true, y_predict, average='macro')
          cm = confusion_matrix(y_true, y_predict)

          ax= plt.subplot()
          sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

          # labels, title and ticks
          ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
          ax.set_title('Confusion Matrix');
          ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
          plt.show()

          # print(i)
          acs_array_smote[i] = acc
          gm_array_smote[i] = gm
          f1macro_array_smote[i] = f1_macro
          precision_smote[i] = precision
          recall_smote[i] = recall

      Accuracy[repeats] = acs_array_smote.mean()
      GeometricMean[repeats] = gm_array_smote.mean()
      F1Score[repeats] = f1macro_array_smote.mean()
      Precision[repeats] = precision_smote.mean()
      Recall[repeats] = recall_smote.mean()

      result_df.loc[len(result_df.index)] = [Accuracy[repeats],GeometricMean[repeats],F1Score[repeats], Precision[repeats], Recall[repeats]]

      # fig_2, ax_2 = plt.subplots()
      fig_2, ax_2 = plt.subplots(figsize=(15, 15))
      fig_2.patch.set_visible(False)
      ax_2.axis('off')
      ax_2.axis('tight')
      ax_2.table(cellText=result_df.values, colLabels=result_df.columns, loc='center')
      fig_2.tight_layout()
      plt.show()

      file_name = 'Metrics_{}_{}.csv'.format(args['data_name'], args['oversampling_method'])
      dst = '/content/drive/MyDrive/Colab/' + args['data_name'] + '/' + file_name
      result_df.to_csv(file_name, mode='a',index=False, header=args['header'])
      shutil.copyfile(file_name, dst)