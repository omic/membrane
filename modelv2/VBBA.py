#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
from utils import *
from network import *

def fwd_pass(user_stfts):
    """
    recordings is the result of split recordings
    returns mean embedding of recordings
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


def load_speaker_models():
    if not os.path.exists(SPEAKER_MODELS_FILE):
        return dict()

    with open(SPEAKER_MODELS_FILE, 'rb') as fhand:
        speaker_models = pickle.load(fhand)

    return speaker_models


def show_current_users():
    speaker_models = load_speaker_models()
    return list(speaker_models.keys())


def get_emb( enroll = False):#fpath
#     record(fpath, enroll)
    denoised_data = record_and_denoise( enroll)
#     user_stfts = split_recording(fpath)
    user_stfts = split_loaded_data(denoised_data, RATE)
    user_stfts = np.expand_dims(user_stfts, axis=1)
#     print(user_stfts.shape)
    emb = fwd_pass(user_stfts)
#     print('emb shape:', emb.shape) #Let's check shape
    return emb, denoised_data  #audio_buffer, bg_buffer 


def emb_dist(emb1, emb2):
    return 1 - scipy.spatial.distance.cdist(emb1, emb2, DISTANCE_METRIC).item()


def enroll_new_user(username):
    fpath = os.path.join(ENROLLMENT_FOLDER, username + '_' + ENROLL_RECORDING_FNAME)
    emb, denoised_data = get_emb( enroll = True)#fpath,
    store_user_embedding(username, emb)
    write_recording(fpath,denoised_data)

def verify_user(username):
#     fpath = os.path.join(VERIFICATION_FOLDER, username + '_' + VERIFY_RECORDING_FNAME)
    #username + '_' + VERIFY_RECORDING_FNAME
#     while os.path.exists(fpath):
#         fname = fpath.split('.wav')[0]
#         if fname[-1].isalpha():
#             fpath = fname+'2'+'.wav'
#         else:
#             fpath = fname[:-1]+str(int(fname[-1])+1)+'.wav'
    emb,  denoised_data = get_emb()#fpath
    speaker_models = load_speaker_models()
#     print(emb.shape, speaker_models[username].shape)  ##let's check the shapes
    dist = emb_dist(emb, speaker_models[username])
    print('cosine distance: ',dist)
    return dist > THRESHOLD , denoised_data   #, fpath


# def fname_numbering(fpath):
#     while os.path.exists(fpath):
#         fname = fpath.split('.wav')[0]
#         if fname[-1].isalpha():
#             fpath = fname+'2'+'.wav'
#         else:
#             fpath = fname[:-1]+str(int(fname[-1])+1)+'.wav'

def identify_user():
#     fpath = os.path.join(VERIFICATION_FOLDER, IDENTIFY_RECORDING_FNAME)
    emb,  denoised_data = get_emb(fpath)
    speaker_models = load_speaker_models()
    dist = [(other_user, emb_dist(emb, speaker_models[other_user]))
            for other_user in speaker_models]
    print('cosine distance: ',dist)
    username, min_dist = min(dist, key=lambda x:x[1])

    if min_dist > THRESHOLD:
        return username,   denoised_data
    return None,  denoised_data


def delete_user(username):
    speaker_models = load_speaker_models()
    _ = speaker_models.pop(username)
    print("Successfully removed {} from database".format(username))
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(speaker_models, fhand)


def clear_database():
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(dict(), fhand)
    print("Deleted all users in database")

def do_list():
    users_list = show_current_users()
    if not users_list:
        print("No users found")
    else:
        print("\n".join(users_list))
        
        
def do_enroll(username):
    assert username is not None, "Enter username"
    if username in show_current_users():
        print("Username already exists in database.")
        var = input("Do you want to replace? (y/n):")
        if var == 'y' or 'yes': pass
        else: return
    enroll_new_user(username)
    
    
def do_verify(username):
    assert username is not None, "Enter username"
    assert username in show_current_users(), "Unrecognized username"
    verified,  denoised_data = verify_user(username)
    if verified:
        print("User verified")
    else:
        print("Unknown user")
    var = input("Save recording: (y/n)?")
    if var == 'y' or var == 'yes':
        fpath = os.path.join(VERIFICATION_FOLDER, username + '_' + VERIFY_RECORDING_FNAME)
        fpath = fpath_numbering(fpath)
        write_recording(fpath,  denoised_data)
#             write_recording(fpath+'_bg.wav', bg_buffer)
        print(f'{fpath}.wav saved')
    else:#if var == 'n' or var == 'no':
#             os.remove(fpath)
#             print(f'{fpath}.wav removed')
        print('Recording removed')
    
    
def do_identify(username):
    identified_user,  denoised_data = identify_user()
    print("Identified User {}".format(identified_user))
    correct_user = input(f"Are you {identified_user}? (y/n): ")

    var = input("Save recording? (y/n): ")
    if var == 'y' or var == 'yes':
        fpath = os.path.join(VERIFICATION_FOLDER, IDENTIFY_RECORDING_FNAME)
        fpath = fpath_numbering(fpath)
        path_split = fpath.rsplit('/',1)
        if correct_user =='y' or correct_user == 'yes':
#                 dir_path = path_split[0]
#                 fname = path_split[-1]
            new_fpath = os.path.join(path_split[0],identified_user+'_'+path_split[-1])
#                 os.rename(fpath, new_fpath)
        else:
            new_fpath = os.path.join(path_split[0],'unknown'+'_'+path_split[-1])
#                 os.rename(fpath, new_fpath)
        write_recording(new_fpath,  denoised_data)
#             write_recording(new_fpath+'_bg.wav', bg_buffer)
        print(f'{new_fpath}.wav saved')

    else: #if var == 'n' or var == 'no':
#             os.remove(fpath)
#             print(f'{fpath} removed')
        print('Recording removed')

    
def do_delete(username):
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

    args = parser.parse_args()

    if args.list:
        do_list()

    elif args.enroll:
        username = args.username
        do_enroll(username)
        
    elif args.verify:
        username = args.username
        do_verify(username)
        

    elif args.identify:
        do_identify()

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
            print("\n".join(users_list))


if __name__ == "__main__":
    main()

