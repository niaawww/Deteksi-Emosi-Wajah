import os

paths_to_try = [
    os.path.join('model', 'emotion_model.h5'),
    os.path.join(os.getcwd(), 'model', 'emotion_model.h5'),
]

for p in paths_to_try:
    print(p, '->', os.path.exists(p))

