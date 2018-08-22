if __name__=="__main__":
	import pandas as pd
	import numpy as np
	data = pd.read_csv("sphist.csv")
	data['Date']=pd.to_datetime(data['Date'])
	from datetime import datetime
	data = data.sort_values("Date")
	print(data.info())
	
	data['avg_5']=data['Close'].rolling(5).mean().shift(1)
	data['avg_30'] = data['Close'].rolling(30).mean().shift(1)
	data['std_5'] = data['Close'].rolling(5).std().shift(1)
	data['std_30'] = data['Close'].rolling(30).std().shift(1)
	data['avg_365'] = data['Close'].rolling(365).std().shift(1)
	data['std_365'] = data['Close'].rolling(365).std().shift(1)
	data['avg_5_365'] = data['avg_5']/data['avg_365']
	data['avg_5_30'] = data['avg_5']/data['avg_30']
	data['std_5_365'] = data['std_5']/data['std_365']
	data['std_5_30'] = data['std_5']/data['std_30']
	#print(data['std_30'].head())
	data=data[data['Date']>=datetime(year=1951,month=1,day=3)]
	data = data.dropna(axis=0)
	
	train = data[data['Date']<datetime(year=2013,month=1,day=1)]
	test = data[data['Date']>=datetime(year=2013,month=1,day=1)]

	print(data.shape)
	print(train.shape)
	print(test.shape)
	
	from sklearn.metrics import mean_squared_error,mean_absolute_error
	from sklearn.linear_model import LinearRegression
	
	target ='Close'
	features = data.columns[7:]
	print(features)
	
	model = LinearRegression().fit(train[features],train[target])
	predicted = model.predict(test[features])
	mse = mean_squared_error(test[target],predicted)
	mae = mean_absolute_error(test[target],predicted)
	rmse = mse**0.5
	print("Mean Squared Error is {}".format(mse))
	print("Mean Absolute Error is {}".format(mae))
	print("Root MSE is {}".format(rmse))
