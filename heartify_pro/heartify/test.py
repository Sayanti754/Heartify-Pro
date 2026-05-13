import joblib
import librosa
import numpy as np

model = joblib.load('heart_model.pkl')
print('Model expects:', model.n_features_in_, 'features')
print('Classes:', model.classes_)

y, sr = librosa.load('New_N_124.wav', sr=22050, mono=True)
y, _ = librosa.effects.trim(y, top_db=20)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50, n_bands=6)
feats = np.concatenate([np.mean(mfcc,axis=1), np.std(mfcc,axis=1), np.mean(chroma,axis=1), np.mean(contrast,axis=1)])
print('Extracted:', feats.shape[0], 'features')
print('Match:', feats.shape[0] == model.n_features_in_)