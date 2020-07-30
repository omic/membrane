#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
from argparse import ArgumentParser
import sys
sys.path.append('./model/modelv2')
# from utils import * ###imported by network
from network import *
from deepspeech import Model#, printVersions

#############Voice-To-Text#############
def store_user_phrase(username:str, phrase:str):
    """
    This function adds 'username' and user's secret 'phrase' into database.
    """
    speaker_phrases = load_speaker_phrases()
    speaker_phrases[username] = phrase
    with open(SPEAKER_PHRASES_FILE, 'wb') as fhand:
        pickle.dump(speaker_phrases, fhand)
    print("Successfully added user {}'s phrase to database".format(username))

def load_speaker_phrases(file = SPEAKER_PHRASES_FILE):
    """
    Load enrolled phrases saved in 'file'.
    """
    if not os.path.exists(file):
        return dict()
    with open(file, 'rb') as fhand:
        speaker_phrases = pickle.load(fhand)
    return speaker_phrases

def identify_user_by_phrase(data, verbose = True):
    """
    Identify the user by matching phrase to enrolled phrases.

    data: audio data as in array.
    verbose: whether to print trascribed phrase.
    """
    phrase = get_text(data)
    speaker_phrases = load_speaker_phrases()
    text_scores = list(map(get_text_score, [phrase]*len(speaker_phrases), speaker_phrases.values()))
    max_idx = np.argmax(text_scores)
    if verbose:
        print('transcribed phrase:', phrase)
        print('phrase scores:',text_scores)
    matched_user = list(speaker_phrases)[max_idx]
    return matched_user

#######Deepspeech Voice-To-Text Parameters########
DS_FOLDER = 'deepspeech_data'
if not os.path.exists(DS_FOLDER):
    os.mkdir(DS_FOLDER)
DS_model_file_path = 'deepspeech_data/deepspeech-0.7.0-models.pbmm'
beam_width = 500
DS_model = Model(DS_model_file_path)
DS_model.setBeamWidth(beam_width)
DS_model.enableExternalScorer('deepspeech_data/deepspeech-0.7.0-models.scorer')

def get_text(data, model = DS_model):
    """
    Transcribe text from audio.

    data: audio data as in array read from librosa with sampling rate 16000.
    model: Deepspeech ASR model.
    """
#     y , s = librosa.load(fpath, sr=16000)
    y = (data* 32767).astype('int16')
    text = model.stt(y)
    return text

def get_text_score(phrase1:str, phrase2:str):
    """
    Return sentence similarity score using SequenceMatcher from difflib.
    """
    return SequenceMatcher(a= phrase1, b= phrase2).ratio()



##############Voice Recognition##########

def fwd_pass(user_stfts):
    """
    recordings is the result of split recordings.
    returns mean embedding of recordings.

    user_stfts: stft array.
    """
    checkpoints = os.listdir(get_rel_path('checkpoints/'))
    checkpoints.sort()
    model, *_ = load_saved_model(checkpoints[-1]) #MODEL_FNAME
    user_stfts = torch.tensor(user_stfts).to(device)
    out = model.forward_single(user_stfts)
    out_np = out.detach().cpu().numpy()
    return np.expand_dims(np.mean(out_np, axis=0), axis=0)


def store_user_embedding(username:str, emb):
    """
    this function adds 'username' and its 'emb' into database.
    emb is mean embedding of the recording returned from fwd_pass.

    emb: mean embedding vector array
    """
    speaker_models = load_speaker_models()
    speaker_models[username] = emb
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(speaker_models, fhand)
    print("Successfully added user {} to database".format(username))


def get_user_embedding(usernames:str):
    """
    returns list of users emb from the db.
    """
    speaker_models = load_speaker_models()
    return [speaker_models[username] for username in usernames]


def load_speaker_models(file = SPEAKER_MODELS_FILE):
    """
    Load enrolled embeddings saved in 'file'.
    """
    if not os.path.exists(file):
        return dict()
    with open(file, 'rb') as fhand:
        speaker_models = pickle.load(fhand)
    return speaker_models


def show_current_users():
    """
    returns list of usernames.
    """
    speaker_models = load_speaker_models()
    return list(speaker_models.keys())

def get_emb( enroll = False, file = '', phrase = ''):
    """
    returns an embedding vector and denoised audio data array.

    file: path to the audio file
        if given, speaker's audio is read from 'file'.
            Miminum of either NOISE_DURATION_FROM_FILE or the first two seconds (RATE*2) will be considered as background noise.
        if not given, invoke record_and_denoise function.
    enroll: indicate whether the user is enrolling or not.
    phrase: phrase is passed if the user provide it. Otherwise pass '' and it will be transcribed later.
    """
    if file:
        data , _ = librosa.load(file,sr=RATE)
        NOISE_DURATION_FROM_FILE = int(len(data)*0.25) # N_D_F_F in terms of lenth of data not second
        NOISE_DURATION_FROM_FILE = min(NOISE_DURATION_FROM_FILE, RATE*2)
        noise, data = np.split(data,[NOISE_DURATION_FROM_FILE])
        denoised_data = removeNoise(data,noise).astype('float32')
    else:
        denoised_data = record_and_denoise( enroll, phrase = '')
    user_stfts = split_loaded_data(denoised_data, RATE)
    user_stfts = np.expand_dims(user_stfts, axis=1)
    emb = fwd_pass(user_stfts)
    return emb, denoised_data

def enroll_new_user(username:str, file = ''):
    """
    Enroll a new user.

    username: user's username for the system.
    file: path to a user's audio file to be used for the enrollment.
        users can use their existing file by passing a path to the file. Otherwise recording function will be invoked.
    """
    if file:
        emb, denoised_data = get_emb( enroll = True, file = file)
        print("\n Please type your phrase.\n")
        print(" If you want to use auto detection of your phrase please hit \'Enter\'.\n")
        phrase = input(" Type phrase or hit \'Enter\': ").lower()
        if phrase == '':
            phrase = get_text(denoised_data)
    else:
        fpath = os.path.join(ENROLLMENT_FOLDER, username + '_' + ENROLL_RECORDING_FNAME)
        print(" \nPlease type a phrase you want to use. \n")
        print(" If you want to use auto detection of your phrase please hit \'Enter\'.\n")
        phrase = input(" Type phrase or hit \'Enter\': ").lower()
        emb, denoised_data = get_emb( enroll = True, phrase = phrase)#fpath,
        write_recording(fpath,denoised_data)
        if phrase =='':
            phrase = get_text(denoised_data)
    store_user_embedding(username, emb)
    store_user_phrase(username, phrase)

def verify_user( file = '', verbose = True):
    """
    Verify user's voice.

    file: path to a user's audio file to be used for the verification.
        users can use their existing file by passing a path to the file. Otherwise recording function will be invoked.
    verbose: whether to print scores.
    """
    if file:
        emb,  denoised_data = get_emb(file = file)
    else:
        emb,  denoised_data = get_emb()#fpath
    speaker_models = load_speaker_models()
    username = identify_user_by_phrase(denoised_data, verbose)
    c_score = cosine_similarity(emb, speaker_models[username])
    E_dist = euclidean_distances(emb, speaker_models[username])
    if verbose:
        print('cosine distance: ',c_score)
        print('Euclidean distance: ',E_dist)
    return (c_score > C_THRESHOLD)and(E_dist < E_THRESHOLD) , denoised_data, username  #, fpath

def identify_user(file = ''):
    """
    -Administrator mode-
    Identify the speaker.

    file: path to a user's audio file to be used for the identification.
        users can use their existing file by passing a path to the file. Otherwise recording function will be invoked.
    """
    if file:
        emb,  denoised_data = get_emb(file = file)
    else:
        emb,  denoised_data = get_emb()#fpath
    speaker_models = load_speaker_models()
    dist = [(other_user, euclidean_distances(emb, speaker_models[other_user]))
            for other_user in speaker_models]#
    print('Euclidean distance: ',dist)
    username, min_distance = min(dist, key=lambda x:x[1])

    if min_distance < E_THRESHOLD:
        return username,   denoised_data
    return None,  denoised_data

def delete_user(username:str):
    """
    -Administrator mode-
    Delete 'username' from the enrollment files.
    """
    speaker_models = load_speaker_models()
    _ = speaker_models.pop(username)
    speaker_phrases = load_speaker_phrases()
    _ = speaker_phrases.pop(username)
    print("Successfully removed {} from database".format(username))
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(speaker_models, fhand)
    with open(SPEAKER_PHRASES_FILE, 'wb') as fhand:
        pickle.dump(speaker_phrases, fhand)

def clear_database():
    """
    -Administrator mode-
    Delete all enrolled users.
    """
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(dict(), fhand)
    print("Deleted all users in database")

def do_list():
    """
    -Administrator mode-
    Print enrolled usernames.
    """
    users_list = show_current_users()
    if not users_list:
        print("No users found")
    else:
        print("\n".join(users_list))

def do_enroll(username:str, file = ''):
    """
    Invoke enroll_new_user function with instructions.

    file: path to an audio file if a user wants to use an existing file.
    """
    print()
    assert username is not None, "Enter username"
    if username in show_current_users():
        print("Username already exists in database.")
        var = input("Do you want to replace? (y/n):")
        if var == 'y' or var =='yes':
            pass
        else:
            return
    enroll_new_user(username, file = file)

def do_verify( file = ''):
    """
    Invoke verify_user function with instructions.

    file: path to an audio file if a user wants to use an existing file.
    """
    print()
    verified,  denoised_data, username = verify_user( file = file, verbose = True)
    if verified:
        print("User verified: ", username)
    else:
        print("Unknown user")
    var = input("Save recording: (y/n)?")
    if var == 'y' or var == 'yes':
        fpath = os.path.join(VERIFICATION_FOLDER, username + '_' + VERIFY_RECORDING_FNAME)
        fpath = fpath_numbering(fpath)
        write_recording(fpath,  denoised_data)
        print(f'{fpath}.wav saved')
    else:
        print('Recording removed')

def do_identify( file = ''):
    """
    Invoke identify_user function with instructions.

    file: path to an audio file if a user wants to use an existing file.
    """
    identified_user,  denoised_data = identify_user(file = file)
    print("Identified User {}".format(identified_user))
    correct_user = input(f"Are you {identified_user}? (y/n): ")

    var = input("Save recording? (y/n): ")
    if var == 'y' or var == 'yes':
        fpath = os.path.join(VERIFICATION_FOLDER, IDENTIFY_RECORDING_FNAME)
        fpath = fpath_numbering(fpath)
        path_split = fpath.rsplit('/',1)
        if correct_user =='y' or correct_user == 'yes':
            new_fpath = os.path.join(path_split[0],identified_user+'_'+path_split[-1])
        else:
            new_fpath = os.path.join(path_split[0],'unknown'+'_'+path_split[-1])
        write_recording(new_fpath,  denoised_data)
        print(f'{new_fpath}.wav saved')
    else:
        print('Recording removed')

def do_delete(username:str):
    """
    Invoke delete_user function with instructions.
    """
    assert username is not None, "Enter username"
    assert username in show_current_users(), "Unrecognized username"
    delete_user(username)

def main():
    running = True
    file = ''
    while running:
        args = input("\n Please type \'enroll\' or \'e\' to enroll a new user,\n  type \'verify\' or \'v\' to verify an enrolled user:").lower()
        print()
        if args == 'enroll' or args == 'e':
            username = input(" Please type your username:")
            do_enroll(username, file)
            running = False
        elif args == 'verify' or args =='v':
            do_verify(file)
            running = False
        elif args == 'd' or args == 'delete':
            username = input(" Please type username to delete:")
            do_delete(username)
            running = False
        elif args == 'c' or args == 'clear':
            clear_database()
        elif args == 'f' or args == 'file':
            file = input(' Please input file path:')
        else:
            print(' Please enter "e" or "v".')

if __name__ == "__main__":
    main()

