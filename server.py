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
"""

import json
import subprocess

from flask import Flask, request, jsonify

from membrane import *

app = Flask(__name__)

# TODO:  Start storing this data in S3, not the EC2.
INDEX_PATH = '~/membrane/users/index.json'
SAMPLE_PATH = '~/membrane/users/samples'

def _read_db(email: str) -> dict:
    """Get user phrase and location to all samples files pertaining
    to this user.
    """
    try:
        with open(INDEX_PATH, 'rt') as user_index:
            idx = json.load(user_index)
            entry = idx[email]
            # TODO:  Return filenames and secret phrase for user with this 
            #        email
    except:
        return None

def _write_db(email: str, phrase: str = None, sample: bytes = None) -> None:
    """Write either phrase or new sample to user entry in index."""
    if sample:
        # TODO:  Write to SAMPLE_PATH and get path to file.
        sample_path = '...'
    with open(INDEX_PATH, 'rt+') as user_index:
        idx = json.load(user_index)
        entry = {}
        if sample_path:
            # TODO:  Update sample path to entry.
            pass
        if phrase:
            # TODO:  Add phrase to entry.
            pass
        json.dump(idx, user_index)

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
        print('User verified: ', username)
        return 1
    else:
        print('Unknown user')
        return 0

@app.route('/enroll', methods=['POST'])
def enroll():
    """Train new user."""
    username = my_form_post()
    assert username is not None, 'Enter username'
    if username in show_current_users():
        print('Username already exists in database.')
        # var = input('Do you want to replace? (y/n):')
        var = my_form_post()
        if var == 'y' or var =='yes':
            pass
        else:
            return
    enroll_new_user(username)

    #     text = request.form['text']
    #     processed_text = text.lower()
    #     return processed_text

        ### for test version, ask if we can save their audio ###
    #     # var = input('Save recording: (y/n)?')
    #     var = my_form_post()
    #     if var == 'y' or var == 'yes':
    #         fpath = os.path.join(VERIFICATION_FOLDER, username + '_' + VERIFY_RECORDING_FNAME)
    #         fpath = fpath_numbering(fpath)
    #         write_recording(fpath,  denoised_data)
    # #             write_recording(fpath+'_bg.wav', bg_buffer)
    #         print(f'{fpath}.wav saved')
    #     else:#if var == 'n' or var == 'no':
    # #             os.remove(fpath)
    # #             print(f'{fpath}.wav removed')
    #         print('Recording removed')

# Liftoff!
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int('5000'), debug=True)