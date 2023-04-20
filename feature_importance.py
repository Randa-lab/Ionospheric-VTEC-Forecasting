import matplotlib.pyplot as plt

features = data_df.columns[:-2]

importance = model_dtree.feature_importances_
  
  # summarize feature importance
  for i,v in enumerate(importance):
    print('Feature: %0d, Relative importance: %.5f' % (i,v))

  # plot feature importance
  plt.barh([x_train for x_train in range(len(importance))], importance, align='center')
  plt.yticks(range(len(features)), features)
  plt.xticks((np.arange(0.0, 1.1, 0.2)))
  plt.rcParams ['figure.figsize'] = [5, 3]
  plt.rcParams.update({'font.size': 14})
  plt.xlabel('Relative importance')
  plt.show()
