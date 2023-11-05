from Functions.packages_basics import *
from Functions.packages_sklearn import *
from Functions.packages_tf import *

from Functions.func_lstm import func_model, create_X, create_y_0, func_lstm_forecast
from Functions.func_files import func_save_model, func_save_fig, func_save_data

from Functions.settings import *

rLRp = ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=10,
    verbose=1,
    mode='min',
    # min_delta -- threshold for measuring the new optimum, to only focus on significant changes.
    min_delta=0.0001,  # originally, use 0.0001 #
    min_lr=1e-15
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor='loss',
    restore_best_weights=True)

## --------------------- settings ------------------------
n = 0

locs = [408, 415, 422]
ps = [16, 14, 14]
N, p = locs[n], ps[n]
nm_tem = str(N) + "_p" + str(p)

gap, n_lastPoints = 7, 28

## --------------- read  smoothing data -----------------
dat_tem = pd.read_csv(os.path.join(
    path_dataSmooth, nm_tem + "_smooth.csv"), index_col=0).iloc[:, 0].to_numpy()
## ----------------- prepare data -----------------------
scaler_1 = StandardScaler()
dat_tem = dat_tem.reshape(-1, 1)
X_std = scaler_1.fit_transform(dat_tem)

XX, yy = create_X(X_std), create_y_0(X_std)
XX_train, yy_train = XX, yy  # [:-50]
XX_test, yy_test = XX[-50:], yy[-50:]

## ------------------------ model fitting --------------------------------
tf.random.set_seed(1014)
n_features = 1

model1 = func_model(n_features)
history1 = model1.fit(XX_train, yy_train, epochs=3, shuffle=False,
                      verbose=1, batch_size=10, callbacks=[rLRp, early_stopping_cb],
                      validation_data=(XX_test, yy_test))

loss = np.round(history1.history['loss'][-1], 4)
val_loss = np.round(history1.history["val_loss"][-1], 4)
print(loss)
print(val_loss)

y_pred_tr = scaler_1.inverse_transform(model1.predict(XX))

# plt.plot(dat_tem)
# plt.plot(np.arange(gap, len(dat_tem)), y_pred_tr)
# plt.show()

## save model
# model_name = nm_tem
# func_save_model(model1, model_name, path_model)

####  ----------------------- read model -------------------------------
# model_name = nm_tem
# model1 = tf.keras.models.load_model(os.path.join(path_model, model_name) + ".h5")
## ------------------------ forecast ---------------------------------
y_pred_tr = scaler_1.inverse_transform(model1.predict(XX))
y_forecast_tr = func_lstm_forecast(model1, scaler_1, yy_test)
forcast_pd = pd.DataFrame({nm_tem + "forecast": y_forecast_tr})

data_name = nm_tem + "forecast"
func_save_data(forcast_pd, data_name, path_dataForecast)

plt.plot(dat_log, color="black", linewidth=0.7)
plt.plot(np.arange(gap, len(dat_tem)), y_pred_tr, color="brown")
plt.plot(dat_tem, color="blue")
plt.plot(range(N, N+28), y_forecast_tr[:28], color="red", )

fig_name = nm_tem + "_forecast"
# func_save_fig(fig_name, path_fig)
plt.show()
