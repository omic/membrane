#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

# from argparse import ArgumentParser
from utils import *
from network import *


C_THRESHOLD = THRESHOLD = 0.99 # 0.8 # similarity should be larger than
E_THRESHOLD = 3 #distance should be less than

#############Voice-To-Text#############


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
        denoised_data = removeNoise(data,noise)
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

def do_delete(username):
    assert username is not None, "Enter username"
    assert username in show_current_users(), "Unrecognized username"
    delete_user(username)






