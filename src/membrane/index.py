#!/usr/bin/env python3
# -.- coding: utf-8 -.-
"""Simple Membrane server -- voice-based biometric login in your browser.

We support two types of communication:

1.  Enrollment -- training our model to verify a new user.
2.  Verification -- performing verification inference on an existing user
    given their voice and user email.

Ideal project structure:

    ~
        membrane/
            __init__.py
            server.py
            model/
                __init__.py
                ...
            users/
                index.json
                samples/
                    ...

Existing demo is here:  http://34.211.144.112:8501/.
"""

import json
import subprocess

from flask import Flask, request, jsonify

import membrane.main as membrane
import util

app = Flask(__name__)

# FILE and FOLDER PATH variables defined in utils.py
# Copied here
SPEAKER_MODELS_FILE = 'speaker_models.pkl'
SPEAKER_PHRASES_FILE = 'speaker_phrases.pkl'
ENROLLMENT_FOLDER = "enrolled_users" #to save enrolled audio samples (not neccessary)
VERIFICATION_FOLDER = "tested_users" #to save tested audio samples (not neccessary)

# TODO:  Start storing this data in S3, not the EC2.
S3_SERVER = True
S3_PATH = '...'
if S3_SERVER:
    from smart_open import open
def get_rel_path(path, server=S3_SERVER, root_dir=S3_PATH):
    if server:
        return os.path.join(root_dir, path)#TODO: amend
    else:
        return path

DATA_PATH = 'data/'
SPEAKER_PHRASES_FILE = get_rel_path(os.path.join(DATA_PATH, SPEAKER_PHRASES_FILE))
SPEAKER_MODELS_FILE = get_rel_path(os.path.join(DATA_PATH, SPEAKER_MODELS_FILE))
ENROLLMENT_FOLDER = get_rel_path(os.path.join(DATA_PATH, ENROLLMENT_FOLDER))
VERIFICATION_FOLDER = get_rel_path(os.path.join(DATA_PATH, VERIFICATION_FOLDER))

if not os.path.exists(ENROLLMENT_FOLDER):
    os.mkdir(ENROLLMENT_FOLDER)

if not os.path.exists(VERIFICATION_FOLDER):
    os.mkdir(VERIFICATION_FOLDER)

if not os.path.exists(COMMENTS_FOLDER):
    os.mkdir(COMMENTS_FOLDER)

def load_speaker_phrases(file = SPEAKER_PHRASES_FILE)->dict:
    if not os.path.exists(file):
        return dict()
    with open(file, 'rb') as fhand:
        speaker_phrases = pickle.load(fhand)
    return speaker_phrases

def load_speaker_models(file = SPEAKER_MODELS_FILE):
    if not os.path.exists(file)->dict():
        return dict()
    with open(file, 'rb') as fhand:
        speaker_models = pickle.load(fhand)
    return speaker_models

def store_user_phrase(username: str, phrase: str, file = SPEAKER_PHRASES_FILE) -> None:
    """
    This function adds username and user's secret phrase into database.
    Will use an email address for a username.
    """
    speaker_phrases = load_speaker_phrases()
    speaker_phrases[username] = phrase
    with open(file, 'wb') as fhand:
        pickle.dump(speaker_phrases, fhand)
    #TODO: UX print("Successfully added user {}'s phrase to database".format(username))


def store_user_embedding(username: str, emb: str) -> None:
    """
    This function adds username(email) and its emb into database.
    emb is mean embedding of the recording returned from fwd_pass.
    In membrane we store an emb vector, not audio itself.
    """
    speaker_models = load_speaker_models()
    speaker_models[username] = emb
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(speaker_models, fhand)
    #TODO: UX print("Successfully added user {} to database".format(username))

###################### READING and WRITING functions upto here #######################
#############################################################################

def record_and_denoise( enroll = False, phrase = ''):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    if enroll:
        #TODO: UX - input(' Ready to start? (press enter)')
    else:
        #TODO: UX - print(" Recording starts soon...\n")

    frames_bg = []
    for i in range(0, int(RATE / CHUNK * (BACKGROUND_RECORD_SECONDS+1) ) ):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames_bg.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    #TODO: UX - print(" Recording starts in 3 second...")
    time.sleep(2)   # start 1 second earlier
    frames = []
    #TODO: UX - print(" Speak now!")
    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    #TODO: UX - print(" Recording complete.")
    audio_data = (np.frombuffer(b''.join(frames), dtype=np.int16)/32767)
    bg_data = (np.frombuffer(b''.join(frames_bg), dtype=np.int16)/32767)
    denoised_data = removeNoise(audio_data, bg_data).astype('float32')
    return denoised_data

def enroll_new_user(username, file = '', phrase = ''):
    if file:
        emb, denoised_data = get_emb( enroll = True, file = file)
        if phrase == '':
            phrase = get_text(denoised_data)
    else:
        #TODO: UX - print(" \nPlease type a phrase you want to use. \n")
        #TODO: UX - print(" If you want to use auto detection of your phrase please hit \'Enter\'.\n")
        emb, denoised_data = get_emb( enroll = True, phrase = phrase)
        #######TODO if want to save audio file ########
        # fpath = os.path.join(ENROLLMENT_FOLDER, username + '_' + ENROLL_RECORDING_FNAME)
        # fpath = fpath_numbering(fpath)
        # write_recording(fpath,denoised_data)
        ################################################
        if phrase =='':
            phrase = get_text(denoised_data)
    store_user_embedding(username, emb)
    store_user_phrase(username, phrase)
    return denoised_data


@app.route('/sanity', methods=['POST'])
def sanity():
    """Allow client to see if they can communicate their audio stream with 
    us.
    """
    if request.method == 'POST':
        user_email = request.args.get('email')
        blob = request.files['file'].read()
        size = len(blob)
        return jsonify({ 'size': size })

@app.route('/verify', methods=['POST'])
def verify():
    """Train new user."""
    verified,  denoised_data, username = verify_user()# file = file)
    if verified:
        #TODO: UX - print('Welcome!')
        return 1
    else:
        #TODO: UX - print('Unknown user')
        #TODO: Send magic link
        return 0

@app.route('/enroll', methods=['POST'])
def enroll():
    """Train new user.
    
    params: {
        'email': '',
        'mode': 'append|write'
        'env': 0
        'phrase': ''
    }
    payload = [byte stream] 
    """
    # Extract fields from payload
    email = request.args['email']
    phrase = request.args.get('phrase', gen_rando_phrase())
    mode = request.args.get('mode', 'write')
    env = request.args.get('env')
    blob = request.files['file'].read()
    if email in show_current_users():
        if mode == 'write':
            # Overwrite existing user 
            pass
        elif mode == 'append':
            # TODO:  Append sample.
            # enroll_existing_user(username, phrase)
            pass
        else:
            raise ValueError(f'Invalid mode {mode}.')
    # TODO:  Ensure `env` doesn't already exist.
    # TODO:  num_samples
    enroll_new_user(username, phrase)
    jsonify(
        email=email,
        phrase=phrase,
        num_samples=num_samples
    )

# Liftoff!
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int('5000'), debug=True)
