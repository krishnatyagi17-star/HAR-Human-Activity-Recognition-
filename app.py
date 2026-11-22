import streamlit as st
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from features import get_features_v2

names = {1: 'Walking', 2: 'Walking Upstairs', 3: 'Walking Downstairs',
         4: 'Sitting', 5: 'Standing', 6: 'Laying'}

model = joblib.load('har_model.pkl')

choice = st.sidebar.radio('Data source', ['UCI test data', 'Our own recordings'])
threshold = st.sidebar.slider('Confidence needed to answer', 0.0, 1.0, 0.8, 0.05)
st.sidebar.write('If the model is less sure than this, it says "Not sure" instead of guessing.')

if choice == 'UCI test data':
    windows = np.load('demo_windows.npy')
    labels = np.load('demo_labels.npy')
else:
    windows = np.load('own_windows.npy')
    labels = np.load('own_labels.npy')


def get_probs(window):
    f = np.array(get_features_v2(window)).reshape(1, -1)
    return model.predict_proba(f)[0]


def label_of(probs):
    return names[model.classes_[probs.argmax()]]


def draw(window):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(window[:, 6], label='x')
    ax.plot(window[:, 7], label='y')
    ax.plot(window[:, 8], label='z')
    ax.set_title('Phone acceleration in this window')
    ax.legend(loc='upper right')
    return fig


def show_prediction(probs):
    guess = label_of(probs)
    conf = probs.max()
    if conf < threshold:
        st.warning('Not sure (best guess: ' + guess + ', only ' + str(round(conf * 100)) + '% sure)')
        return None
    st.subheader('Predicted: ' + guess + ' (' + str(round(conf * 100)) + '% sure)')
    return guess


st.title('Human Activity Recognition Demo')
st.write('Data source: ' + choice + ' (' + str(len(windows)) + ' windows)')

mode = st.radio('Mode', ['Single sample', 'Live replay'])

if mode == 'Single sample':
    idx = st.slider('Sample number', 0, len(windows) - 1, 0)
    actual = names[labels[idx]]
    guess = show_prediction(get_probs(windows[idx]))
    st.write('Actual activity: ' + actual)
    if guess is None:
        st.info('The model chose not to answer this one.')
    elif guess == actual:
        st.success('Correct')
    else:
        st.error('Wrong')
    fig = draw(windows[idx])
    st.pyplot(fig)
    plt.close(fig)

else:
    activity = st.selectbox('Start from the first sample of', list(names.values()))
    smooth = st.checkbox('Smooth predictions (average of the last 3 windows)', value=True)
    key = [k for k, v in names.items() if v == activity][0]
    found = np.where(labels == key)[0]
    start = int(found[0]) if len(found) > 0 else 0
    steps = min(30, len(windows) - start)
    st.write('Starting at sample ' + str(start) + ', showing ' + str(steps) + ' windows')
    if st.button('Start replay'):
        box = st.empty()
        history = []
        raw_right = 0
        smooth_right = 0
        answered = 0
        answered_right = 0
        for i in range(start, start + steps):
            probs = get_probs(windows[i])
            history.append(probs)
            smooth_probs = np.mean(history[-3:], axis=0)
            raw_guess = label_of(probs)
            smooth_guess = label_of(smooth_probs)
            actual = names[labels[i]]
            if raw_guess == actual:
                raw_right += 1
            if smooth_guess == actual:
                smooth_right += 1
            if smooth:
                shown = smooth_probs
            else:
                shown = probs
            with box.container():
                guess = show_prediction(shown)
                st.write('Actual activity: ' + actual)
                st.write('Without smoothing: ' + raw_guess + '  |  With smoothing: ' + smooth_guess)
                fig = draw(windows[i])
                st.pyplot(fig)
                plt.close(fig)
            if guess is not None:
                answered += 1
                if guess == actual:
                    answered_right += 1
            time.sleep(0.6)
        st.write('Replay finished (' + str(steps) + ' windows).')
        st.write('Without smoothing: ' + str(raw_right) + ' correct. With smoothing: ' + str(smooth_right) + ' correct.')
        st.write('The model answered ' + str(answered) + ' windows (' + str(answered_right) + ' correct) and said "Not sure" on ' + str(steps - answered) + '.')
