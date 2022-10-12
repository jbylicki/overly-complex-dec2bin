import numpy as np
from model_def import build_model_func

model = build_model_func()

model.load_weights('./model.h5')
print((model.predict(np.array([[1], [2], [17]])) > 0.5).astype(np.uint8))
print((model.predict(np.array([[1], [2], [17]]))).astype(np.float))
