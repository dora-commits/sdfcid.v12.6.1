file_name = 'Metrics_{}_{}.csv'.format(args['data_name'], args['oversampling_method'])
result = pd.read_csv('/content/drive/MyDrive/Colab/' + args['data_name'] + '/' + file_name, header=[0])
result