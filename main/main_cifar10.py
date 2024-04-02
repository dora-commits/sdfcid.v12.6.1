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

        trainX_imbal, trainY_imbal = make_imbalanced_dataset(trainX, trainY)
        dataset_train = torch.utils.data.TensorDataset(trainX_imbal, trainY_imbal)

        train_AE(dataset_train, i, args['data_name'])

        combx, comby = GenerateSamples(dataset_train, i, args['data_name'])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetClassifier()
        model.to(device)

        combx = combx.reshape(combx.shape[0],3,32,32)
        combx = torch.Tensor(combx)
        comby = torch.tensor(comby,dtype=torch.long)

        x_train, x_validation, y_train, y_validation = train_test_split(combx, comby, test_size=0.2, random_state=0, stratify=comby)

        train_set = TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'],shuffle=True,num_workers=1)

        validation_set = TensorDataset(x_validation, y_validation)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args['batch_size'],shuffle=True,num_workers=1)

        print('Total: '     , combx.shape)
        print('Total class:'      , collections.Counter(comby.numpy()))
        print('train set total: ', x_train.shape)
        print('train set class:' , collections.Counter(y_train.numpy()))
        print('validation set total: '      , x_validation.shape)
        print('validation set class:'       , collections.Counter(y_validation.numpy()))

        # del X_train
        # del X_validation
        # del y_train
        # del y_validation
        # del combx
        # del comby

        train_loss_step, train_acc_step, validation_loss_step, validation_acc_step = model.train_model(train_loader, validation_loader, args['resnet_num_epochs'], args['resnet_learning_rate'])

        train_loss_total.append(train_loss_step)
        train_acc_total.append(train_acc_step)
        validation_loss_total.append(validation_loss_step)
        validation_acc_total.append(validation_acc_step)

        resnet_outputs = model.predict_model(testX.to(device))
        torch.cuda.empty_cache()

        y_predict = [torch.argmax(resnet_outputs[i]).item() for i in range(resnet_outputs.shape[0])]
        y_true = [testY[i].item() for i in range(resnet_outputs.shape[0])]

        acc = accuracy_score(y_true, y_predict)
        gm = geometric_mean_score(y_true, y_predict, average='macro')
        f1_macro = f1_score(y_true, y_predict, average='macro')
        precision = precision_score(y_true, y_predict, average='macro')
        recall = recall_score(y_true, y_predict, average='macro', zero_division=0)
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