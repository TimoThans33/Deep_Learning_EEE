def custom_loss(true_values, predicted_values):
    maxpix = 255
    return K.mean(K.abs(ssim(true_values, predicted_values, maxpix)), axis = -1)

def custom_loss(true_values, predicted_values):
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    covar_true_pred = K.mean(y_true*y_pred, axis=-1) - u_true*u_pred

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)

    ssim /= K.clip(denom, K.epsilon(), np.inf)
    return K.mean(ssim), axis = -1)

def custom_loss(true_values, predicted_values):
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    covar_true_pred = K.mean(y_true*y_pred, axis=-1) - u_true*u_pred

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)

    ssim /= K.clip(denom, K.epsilon(), np.inf)
    return K.mean((1.0-ssim)/2, axis = -1)

def custom_loss(true_values, predicted_values):
    return K.mean(K.square(K.abs(predicted_values - true_values)), axis=-1)

"""
These actually worked:
"""
def custom_loss(true_values, predicted_values):
    maxpix = 255
    return K.mean(K.abs(1-ssim(true_values, predicted_values, maxpix)), axis = -1)

def custom_loss(true_values, predicted_values):
    maxpix = 255
    return K.mean(1/psnr(true_values, predicted_values, maxpix), axis = -1)
