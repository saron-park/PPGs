from scipy import io
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Dense, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling1D

# 데이터 로딩
mat_file = io.loadmat('PPGs_240.mat')

# 전체 데이터 크기 계산
Total_size = 0
for i in range(1, 33):
    t_data = mat_file['split_s' + str(i)]
    for j in range(40):
        k = 0
        while k < len(t_data[j]) and t_data[j][k].size != 0:
            k += 1
        Total_size += k

# 데이터 및 레이블 초기화
data_valence = np.zeros((Total_size, 240, 1), dtype='float32')
data_arousal = np.zeros((Total_size, 240, 1), dtype='float32')
NN_valence = np.zeros((Total_size, 9, 1), dtype='float32')
NN_arousal = np.zeros((Total_size, 9, 1), dtype='float32')
feature_valence = np.zeros((Total_size, 10, 1), dtype='float32')
feature_arousal = np.zeros((Total_size, 10, 1), dtype='float32')
labels_valence = np.zeros(Total_size, dtype='int32')
labels_arousal = np.zeros(Total_size, dtype='int32')

idx_val = 0
idx_arou = 0

# 데이터 추출 및 레이블링
for i in range(1, 33):
    temp_data = mat_file[f'split_s{i}']
    temp_NN = mat_file[f'NNinter_ststis_feat_multi_s{i}']
    temp_labels = mat_file[f'labels_{i}Copy']

    for j in range(40):
        for k in range(len(temp_data[j])):
            if temp_data[j][k].size != 0:
                ppg = temp_data[j][k].reshape(240, 1)
                nn_feat = temp_NN[j][k].reshape(9, 1)
                stat_feat_valence = temp_NN[j + 80][k].reshape(10, 1)
                stat_feat_arousal = temp_NN[j + 40][k].reshape(10, 1)

                # Valence
                if temp_labels[j][0] >= 7:
                    labels_valence[idx_val] = 1
                    data_valence[idx_val] = ppg
                    NN_valence[idx_val] = nn_feat
                    feature_valence[idx_val] = stat_feat_valence
                    idx_val += 1
                elif temp_labels[j][0] <= 3:
                    labels_valence[idx_val] = 0
                    data_valence[idx_val] = ppg
                    NN_valence[idx_val] = nn_feat
                    feature_valence[idx_val] = stat_feat_valence
                    idx_val += 1

                # Arousal
                if temp_labels[j][1] >= 7:
                    labels_arousal[idx_arou] = 1
                    data_arousal[idx_arou] = ppg
                    NN_arousal[idx_arou] = nn_feat
                    feature_arousal[idx_arou] = stat_feat_arousal
                    idx_arou += 1
                elif temp_labels[j][1] <= 3:
                    labels_arousal[idx_arou] = 0
                    data_arousal[idx_arou] = ppg
                    NN_arousal[idx_arou] = nn_feat
                    feature_arousal[idx_arou] = stat_feat_arousal
                    idx_arou += 1

# 데이터 정규화
data_valence /= 1000
data_arousal /= 1000
NN_valence = (NN_valence - 100) / 50
NN_arousal = (NN_arousal - 100) / 50

# 학습 및 테스트 데이터 분할
def split_data(data, nn_data, feature_data, labels, train_ratio=0.8):
    split_idx = int(len(labels) * train_ratio)
    return (data[:split_idx], data[split_idx:],
            nn_data[:split_idx], nn_data[split_idx:],
            feature_data[:split_idx], feature_data[split_idx:],
            labels[:split_idx], labels[split_idx:])

(train_data_valence, test_data_valence, train_NN_valence, test_NN_valence,
 train_feature_valence, test_feature_valence, train_labels_valence, test_labels_valence) = \
    split_data(data_valence[:idx_val], NN_valence[:idx_val], feature_valence[:idx_val], labels_valence[:idx_val])

(train_data_arousal, test_data_arousal, train_NN_arousal, test_NN_arousal,
 train_feature_arousal, test_feature_arousal, train_labels_arousal, test_labels_arousal) = \
    split_data(data_arousal[:idx_arou], NN_arousal[:idx_arou], feature_arousal[:idx_arou], labels_arousal[:idx_arou])

# 레이블 원-핫 인코딩
train_labels_valence = to_categorical(train_labels_valence)
test_labels_valence = to_categorical(test_labels_valence)
train_labels_arousal = to_categorical(train_labels_arousal)
test_labels_arousal = to_categorical(test_labels_arousal)

# PPGs 모델
# 모델 구조 변경
input_PPGs = Input(shape=(240, 1))
x = Conv1D(8, 3, padding='same', activation='relu')(input_PPGs)  # 필터 수 증가
x = Conv1D(8, 3, padding='same', activation='relu')(input_PPGs)  # 필터 수 증가
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(16, 3, padding='same', activation='relu')(x)  # 필터 수 증가
x = Dropout(0.2)(x)  # 드롭아웃 비율 증가
x = Conv1D(16, 3, padding='same', activation='relu')(x)  # 필터 수 증가
x = Dropout(0.2)(x)  # 드롭아웃 비율 증가
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(32, 3, padding='same', activation='relu')(x)  # 필터 수 증가
x = Dropout(0.3)(x)  # 드롭아웃 비율 증가
x = Conv1D(32, 3, padding='same', activation='relu')(x)  # 필터 수 증가
x = Dropout(0.3)(x)  # 드롭아웃 비율 증가
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = GlobalAveragePooling1D()(x)
model_PPGs = Model(inputs=input_PPGs, outputs=x)

# NNinterval 모델
input_NN = Input(shape=(9, 1))
y = Conv1D(8, 3, padding='same', activation='relu')(input_NN)  # 필터 수 증가
y = BatchNormalization()(y)
y = MaxPooling1D(2)(y)
y = Conv1D(8, 3, padding='same', activation='relu')(y)  # 필터 수 증가
y = BatchNormalization()(y)
y = MaxPooling1D(2)(y)
y = GlobalAveragePooling1D()(y)
model_NN = Model(inputs=input_NN, outputs=y)

# 통계 feature 그대로 사용
input_feature = Input(shape=(10, 1))
z = Flatten()(input_feature)
model_feature = Model(inputs=input_feature, outputs=z)

# 통합
combined = concatenate([model_PPGs.output, model_NN.output, model_feature.output])
combined = Dense(128, activation="relu")(combined)
combined = Dropout(0.5)(combined)  # 드롭아웃 비율 증가
output = Dense(2, activation="softmax")(combined)
model_total = Model(inputs=[input_PPGs, input_NN, input_feature], outputs=output)

# 콜백 설정
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

# Valence 모델 학습 및 평가
model_total.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])  # 학습률 조정
history_valence = model_total.fit(
    [train_data_valence, train_NN_valence, train_feature_valence], train_labels_valence,
    epochs=60, batch_size=128, validation_split=0.2,  # 에폭 수 증가
    callbacks=callbacks
)
test_loss_valence, test_acc_valence = model_total.evaluate(
    [test_data_valence, test_NN_valence, test_feature_valence], test_labels_valence, verbose=0
)
print(f"Valence Test Loss: {test_loss_valence:.4f}, Valence Test Accuracy: {test_acc_valence:.4f}")

# Arousal 모델 학습 및 평가 (동일한 모델 사용)
model_total.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])  # 학습률 조정
history_arousal = model_total.fit(
    [train_data_arousal, train_NN_arousal, train_feature_arousal], train_labels_arousal,
    epochs=60, batch_size=128, validation_split=0.2,  # 에폭 수 증가
    callbacks=callbacks
)
test_loss_arousal, test_acc_arousal = model_total.evaluate(
    [test_data_arousal, test_NN_arousal, test_feature_arousal], test_labels_arousal, verbose=0
)
print(f"Arousal Test Loss: {test_loss_arousal:.4f}, Arousal Test Accuracy: {test_acc_arousal:.4f}")
# 학습 곡선 그리기 함수
def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(f'Training and validation accuracy ({title})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss ({title})')
    plt.legend()
    plt.show()

# 학습 곡선 시각화
plot_history(history_valence, 'Valence')
plot_history(history_arousal, 'Arousal')