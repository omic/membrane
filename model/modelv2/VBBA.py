#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
from utils import *
from network import *

sample_phrase_list = [\
"Oak is strong and also gives shade.",\
"Cats and dogs each hate the other.",\
"The pipe began to rust while new.",\
"Open the crate but don't break the glass.",\
"Add the sum to the product of these three.",\
"Thieves who rob friends deserve jail.",\
"The ripe taste of cheese improves with age.",\
"Act on these orders with great speed.",\
"The hog crawled under the high fence.",\
"Move the vat over the hot fire."]



def fwd_pass(user_stfts):
    """
    recordings is the result of split recordings
    returns mean embedding of recordings

    user_stfts: stft array.
    """
    checkpoints = os.listdir(get_rel_path('checkpoints/'))
    checkpoints.sort()
    model, *_ = load_saved_model(checkpoints[-1]) #MODEL_FNAME

    user_stfts = torch.tensor(user_stfts).to(device)
#     print('user_stfts.shape:', user_stfts.shape)  ##let's check the shapes
    out = model.forward_single(user_stfts)
    out_np = out.detach().cpu().numpy()

    return np.expand_dims(np.mean(out_np, axis=0), axis=0)


def store_user_embedding(username, emb):
    """
    this function adds username and its emb into database
    emb is mean embedding of the recording returned from fwd_pass

    emb: mean embedding vector array
    """
    speaker_models = load_speaker_models()
    speaker_models[username] = emb
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(speaker_models, fhand)
    print("Successfully added user {} to database".format(username))


def get_user_embedding(usernames):
    """
    returns list of users emb from the db
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


def get_emb( enroll = False, file = ''):
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
        denoised_data = record_and_denoise( enroll, sample_phrase_list=sample_phrase_list)
    user_stfts = split_loaded_data(denoised_data, RATE)
    user_stfts = np.expand_dims(user_stfts, axis=1)
    emb = fwd_pass(user_stfts)
    return emb, denoised_data


def enroll_new_user(username, file = ''):
    """
    Enroll a new user.

    username: user's username for the system.
    file: path to a user's audio file to be used for the enrollment.
        users can use their existing file by passing a path to the file. Otherwise recording function will be invoked.
    """
    if file:
        emb, denoised_data = get_emb( enroll = True, file = file)
        store_user_embedding(username, emb)
    else:
        fpath = os.path.join(ENROLLMENT_FOLDER, username + '_' + ENROLL_RECORDING_FNAME)
        emb, denoised_data = get_emb( enroll = True)
        store_user_embedding(username, emb)
        write_recording(fpath,denoised_data)

def verify_user(username, file = ''):
    """
    Verify user's voice.

    file: path to a user's audio file to be used for the verification.
        users can use their existing file by passing a path to the file. Otherwise recording function will be invoked.
    """
    if file:
        emb,  denoised_data = get_emb(file = file)
    else:
        emb,  denoised_data = get_emb()#fpath
    speaker_models = load_speaker_models()
#     print(emb.shape, speaker_models[username].shape)  ##let's check the shapes
    c_score = cosine_similarity(emb, speaker_models[username])
    E_dist = euclidean_distances(emb, speaker_models[username])
    print('cosine similarity: ',c_score)
    print('Euclidean distance: ',E_dist)
    return (c_score > C_THRESHOLD)and(E_dist < E_THRESHOLD) , denoised_data

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
    username, min_distance = min(dist, key=lambda x:x[1])
    print('Euclidean distance: ',min_distance, ' to ', username)
    if min_distance < E_THRESHOLD:
        return username,   denoised_data
    return None,  denoised_data


def delete_user(username):
    """
    -Administrator mode-
    Delete 'username' from the enrollment files.
    """
    speaker_models = load_speaker_models()
    _ = speaker_models.pop(username)
    print("Successfully removed {} from database".format(username))
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(speaker_models, fhand)


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


def do_enroll(username, file = ''):
    """
    Invoke enroll_new_user function with instructions.

    file: path to an audio file if a user wants to use an existing file.
    """
    assert username is not None, "Enter username"
    if username in show_current_users():
        print("Username already exists in database.")
        var = input("Do you want to replace? (y/n):")
        if var == 'y' or var =='yes': pass
        else: return
    enroll_new_user(username, file = file)


def do_verify(username, file = ''):
    """
    Invoke verify_user function with instructions.

    file: path to an audio file if a user wants to use an existing file.
    """
    assert username is not None, "Enter username"
    assert username in show_current_users(), "Unrecognized username"
    verified,  denoised_data = verify_user(username, file = file)
    if verified:
        print("User verified")
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


def do_delete(username):
    """
    Invoke delete_user function with instructions.
    """
    assert username is not None, "Enter username"
    assert username in show_current_users(), "Unrecognized username"
    delete_user(username)

def main():
    parser = ArgumentParser(description="Speaker Identification and Verification")
    parser.add_argument('-l', '--list-current-users', dest="list",
                        default=False, action="store_true",
                        help="Show current enrolled users")
    parser.add_argument('-e', '--enroll', dest="enroll",
                        default=False, action="store_true",
                        help="Enroll a new user")
    parser.add_argument('-v', '--verify', dest="verify",
                        default=False, action="store_true",
                        help="Verify a user from the ones in the database")
    parser.add_argument('-i', '--identify', dest="identify",
                        default=False, action="store_true",
                        help="Identify a user")
    parser.add_argument('-d', '--delete', dest="delete",
                        default=False, action="store_true",
                        help="Delete user from database")
    parser.add_argument('-c', '--clear', dest="clear",
                        default=False, action="store_true",
                        help="Clear Database")
    parser.add_argument('-u', '--username', type=str, default=None,
                        help="Name of the user to enroll or verify")
    parser.add_argument('-f', '--with-file', dest='file', default='',
                        help="Provide a recording file rather than record")

    args = parser.parse_args()

    if args.list:
        do_list()

    elif args.enroll:
        username = args.username
        do_enroll(username, args.file)

    elif args.verify:
        username = args.username
        do_verify(username, args.file)


    elif args.identify:
        do_identify(args.file)

    elif args.delete:
        username = args.username
        do_delete(username)


    elif args.clear:
        clear_database()

    else:
        users_list = show_current_users()
        if not users_list:
            print("No users found")
        else:
            print("Users:")
            print("\n".join(users_list))


if __name__ == "__main__":
    main()

