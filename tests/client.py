#!/usr/bin/env python3
# -.- coding: utf-8 -.-
"""Run tests against Membrane server."""

import requests

URL = 'localhost:5000'
AUDIO_FILE = 'luke.m4a'

def test_upload():
    '''
    params: {
        email: "",
        phrase: ""
    }
    payload = [byte stream] 
    '''
    with open(AUDIO_FILE, 'rb') as f:
        audio_bytes = f.read()
        params = {
            'email': '',
            'env': 0,
            'phrase': ''
        }
        requests.post(URL, audio_bytes, )

def main(): 
    test_upload()

if __name__ == '__main__':
    main()