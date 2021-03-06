#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
from utils import *
from network import *

STREAMLIT_PATH = 'streamlit_data/'
SPEAKER_PHRASES_FILE = os.path.join(STREAMLIT_PATH, SPEAKER_PHRASES_FILE)
SPEAKER_MODELS_FILE = os.path.join(STREAMLIT_PATH, SPEAKER_MODELS_FILE)
ENROLLMENT_FOLDER = os.path.join(STREAMLIT_PATH, ENROLLMENT_FOLDER)
VERIFICATION_FOLDER = os.path.join(STREAMLIT_PATH, VERIFICATION_FOLDER)
COMMENTS_FOLDER = os.path.join(STREAMLIT_PATH, 'comments')

C_THRESHOLD = THRESHOLD = 0.99 # 0.8 # similarity should be larger than
E_THRESHOLD = 3 #distance should be less than

if not os.path.exists(ENROLLMENT_FOLDER):
    os.mkdir(ENROLLMENT_FOLDER)

if not os.path.exists(VERIFICATION_FOLDER):
    os.mkdir(VERIFICATION_FOLDER)

if not os.path.exists(COMMENTS_FOLDER):
    os.mkdir(COMMENTS_FOLDER)

#############Voice-To-Text#############
def store_user_phrase(username, phrase):
    """
    this function adds username and user's secret phrase into database
    """
    speaker_phrases = load_speaker_phrases()
    speaker_phrases[username] = phrase
    with open(SPEAKER_PHRASES_FILE, 'wb') as fhand:
        pickle.dump(speaker_phrases, fhand)
    print("Successfully added user {}'s phrase to database".format(username))

def load_speaker_phrases(file = SPEAKER_PHRASES_FILE):
    if not os.path.exists(file):
        return dict()

    with open(file, 'rb') as fhand:
        speaker_phrases = pickle.load(fhand)

    return speaker_phrases

def identify_user_by_phrase(data):
    phrase = get_text(data)
    speaker_phrases = load_speaker_phrases()
    max_idx = np.argmax(list(map(get_text_score, [phrase]*len(speaker_phrases), speaker_phrases.values())))
#     print('phrase scores:',list(map(get_text_score, [phrase]*len(speaker_phrases), speaker_phrases.values())))
    matched_user = list(speaker_phrases)[max_idx]
    return matched_user




##############Voice Recognition##########

def fwd_pass(user_stfts):
    """
    recordings is the result of split recordings
    returns mean embedding of recordings
    """
    checkpoints = os.listdir(get_rel_path('checkpoints/'))
    checkpoints.sort()
#     print(checkpoints[-1])
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


def get_emb( enroll = False, file = '', phrase = ''):#fpath
#     record(fpath, enroll)
    if file:
        data , _ = librosa.load(file,sr=RATE)
#         noise = data[:RATE*NOISE_DURATION_FROM_FILE] # N_D_F_F in terms of second
#         data = data[RATE*NOISE_DURATION_FROM_FILE:]
        NOISE_DURATION_FROM_FILE = int(len(data)*0.25) # N_D_F_F in terms of lenth of data not second
        NOISE_DURATION_FROM_FILE = min(NOISE_DURATION_FROM_FILE, RATE*2)
        noise, data = np.split(data,[NOISE_DURATION_FROM_FILE])
        denoised_data = removeNoise(data,noise).astype('float32')
    else:
        denoised_data = record_and_denoise( enroll, phrase = '')
#     user_stfts = split_recording(fpath)
    user_stfts = split_loaded_data(denoised_data, RATE)
    user_stfts = np.expand_dims(user_stfts, axis=1)
#     print(user_stfts.shape)
    emb = fwd_pass(user_stfts)
#     print('emb shape:', emb.shape) #Let's check shape
    return emb, denoised_data  #audio_buffer, bg_buffer 


# def emb_dist(emb1, emb2):
#     return 1 - scipy.spatial.distance.cdist(emb1, emb2, DISTANCE_METRIC).item()


def enroll_new_user(username, file = '', phrase = ''):
    if file:
        emb, denoised_data = get_emb( enroll = True, file = file)
        if phrase == '':
            phrase = get_text(denoised_data)
    else:
        fpath = os.path.join(ENROLLMENT_FOLDER, username + '_' + ENROLL_RECORDING_FNAME)
        print(" \nPlease type a phrase you want to use. \n")
        print(" If you want to use auto detection of your phrase please hit \'Enter\'.\n")
        emb, denoised_data = get_emb( enroll = True, phrase = phrase)#fpath,
        write_recording(fpath,denoised_data)
        if phrase =='':
            phrase = get_text(denoised_data)
    store_user_embedding(username, emb)
    store_user_phrase(username, phrase)
    return denoised_data
    
def verify_user( file = ''):
    if file:
        emb,  denoised_data = get_emb(file = file)
    else:
        emb,  denoised_data = get_emb()#fpath
    speaker_models = load_speaker_models()
    username = identify_user_by_phrase(denoised_data)
    c_score = cosine_similarity(emb, speaker_models[username])
    E_dist = euclidean_distances(emb, speaker_models[username])
#     print('cosine distance: ',c_score)
#     print('Euclidean distance: ',E_dist)
    return (c_score > C_THRESHOLD)and(E_dist < E_THRESHOLD) , username, denoised_data  # ,denoised_data, fpath


# def fname_numbering(fpath):
#     while os.path.exists(fpath):
#         fname = fpath.split('.wav')[0]
#         if fname[-1].isalpha():
#             fpath = fname+'2'+'.wav'
#         else:
#             fpath = fname[:-1]+str(int(fname[-1])+1)+'.wav'

def identify_user(file = ''):
#     fpath = os.path.join(VERIFICATION_FOLDER, IDENTIFY_RECORDING_FNAME)
    if file:
        emb,  denoised_data = get_emb(file = file)
    else:
        emb,  denoised_data = get_emb()#fpath
    speaker_models = load_speaker_models()
    dist = [(other_user, emb_dist(emb, speaker_models[other_user]))
            for other_user in speaker_models]#actually similarity
    print('cosine distance: ',dist)
    username, max_similarity = max(dist, key=lambda x:x[1])

    if max_similarity > THRESHOLD:
        return username,   denoised_data
    return None,  denoised_data


def delete_user(username):
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
    with open(SPEAKER_MODELS_FILE, 'wb') as fhand:
        pickle.dump(dict(), fhand)
    print("Deleted all users in database")

def do_list():
    users_list = show_current_users()
    if not users_list:
        print("No users found")
    else:
        print("\n".join(users_list))
        
        
# def do_enroll(username, file = ''):
#     print()
#     assert username is not None, "Enter username"
#     if username in show_current_users():
#         print("Username already exists in database.")
#         var = input("Do you want to replace? (y/n):")
#         if var == 'y' or var =='yes':
#             pass
#         else:
#             return
#     enroll_new_user(username, file = file)
    
    
# def do_verify( file = ''):
#     print()
# #     assert username is not None, "Enter username"
# #     assert username in show_current_users(), "Unrecognized username"
#     verified,  denoised_data, username = verify_user( file = file)
#     if verified:
#         print("User verified: ", username)
#     else:
#         print("Unknown user")
#     var = input("Save recording: (y/n)?")
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
    
    
def do_identify( file = ''):
    identified_user,  denoised_data = identify_user(file = file)
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
    
    
    
    
    
    
# def main():
# #     parser = ArgumentParser(description="Speaker Identification and Verification")
# #     parser.add_argument('-l', '--list-current-users', dest="list",
# #                         default=False, action="store_true",
# #                         help="Show current enrolled users")
# #     parser.add_argument('-e', '--enroll', dest="enroll",
# #                         default=False, action="store_true",
# #                         help="Enroll a new user")
# #     parser.add_argument('-v', '--verify', dest="verify",
# #                         default=False, action="store_true",
# #                         help="Verify a user from the ones in the database")
# #     parser.add_argument('-i', '--identify', dest="identify",
# #                         default=False, action="store_true",
# #                         help="Identify a user")
# #     parser.add_argument('-d', '--delete', dest="delete",
# #                         default=False, action="store_true",
# #                         help="Delete user from database")
# #     parser.add_argument('-c', '--clear', dest="clear",
# #                         default=False, action="store_true",
# #                         help="Clear Database")
# #     parser.add_argument('-u', '--username', type=str, default=None,
# #                         help="Name of the user to enroll or verify")
# #     parser.add_argument('-f', '--with-file', dest='file', default='',
# #                         help="Provide a recording file rather than record")

# #     args = parser.parse_args()

# #     if args.list:
# #         do_list()
#     running = True
#     file = ''
#     while running:
#         args = input("\n Please type \'enroll\' or \'e\' to enroll a new user,\n  type \'verify\' or \'v\' to verify an enrolled user:").lower()
#         print()
#         if args == 'enroll' or args == 'e':
#             username = input(" Please type your username:") #args.username
#             do_enroll(username, file)#args.file)
#             running = False

#         elif args == 'verify' or args =='v':
#     #         username = args.username
#             do_verify(file)#args.file)
#             running = False

#     #     elif args.identify:
#     #         do_identify(args.file)

#         elif args == 'd' or args == 'delete':
#             username = input(" Please type username to delete:")
#             do_delete(username)
#             running = False

#         elif args == 'c' or args == 'clear':
#             clear_database()
            
#         elif args == 'f' or args == 'file':
#             file = input(' Please input file path:')

#         else:
#             print(' Please enter "e" or "v".')
# #         users_list = show_current_users()
# #         if not users_list:
# #             print("No users found")
# #         else:
# #             print("\n".join(users_list))


# if __name__ == "__main__":
#     main()

