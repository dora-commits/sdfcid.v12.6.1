if args['pre_trained']:
  for repeats in range(0,num_repeats):

      acs_array_smote     = np.zeros(5)
      gm_array_smote      = np.zeros(5)
      f1macro_array_smote = np.zeros(5)
      precision_smote     = np.zeros(5)
      recall_smote        = np.zeros(5)

      for i in range(0, 5):
          aux_X = foldsX.copy()
          aux_Y = foldsY.copy()

          print('Fold {}'.format(i + 1))

          testX = foldsX[i]
          testY = foldsY[i]

          aux_X.pop(i)
          aux_Y.pop(i)

          trainX = torch.cat(aux_X)
          trainY = torch.cat(aux_Y)

          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          model = ResNetClassifier()
          model.to(device)

          np.printoptions(precision=5,suppress=True)

          modpth = '/content/drive/MyDrive/Colab/' + args['data_name'] + '/model/'
          model_f = []
          model_p = modpth + str(i+10) + '/model_SMOTEWAEGAN_CIFAR10.pth'

          model_f.append(model_p)

          path_model = model_f[0]

          model.load_state_dict(torch.load(path_model), strict=False)

          model.eval()

          t0 = time.time()
          resnet_outputs = model.predict_model(testX.to(device))
          torch.cuda.empty_cache()

          y_predict = [torch.argmax(resnet_outputs[i]).item() for i in range(resnet_outputs.shape[0])]
          y_true = [testY[i].item() for i in range(resnet_outputs.shape[0])]
          t1 = time.time()

          prediction_time[repeats]            = np.round((t1 - t0)/60, 5)
          train_model_time[repeats]           = 0
          generate_samples_time[repeats]      = 0
          train_classification_time[repeats]  = 0

          time_df.loc[len(time_df.index)] = [train_model_time[repeats],generate_samples_time[repeats],train_classification_time[repeats], prediction_time[repeats]]

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
          ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']); ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
          plt.show()

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

      file_name = 'Metrics_{}_{}.csv'.format(args['data_name'], args['oversampling_method'])
      dst = '/content/drive/MyDrive/Colab/' + args['data_name'] + '/' + file_name
      result_df.to_csv(file_name, mode='a',index=False, header=args['header'])
      shutil.copyfile(file_name, dst)