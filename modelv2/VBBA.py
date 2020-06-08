#!/usr/bin/env python3
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


def get_emb(fpath):
    record(fpath)
    user_stfts = split_recording(fpath)
    user_stfts = np.expand_dims(user_stfts, axis=1)
    emb = fwd_pass(user_stfts)
    return emb


def emb_dist(emb1, emb2):
    return 1 - scipy.spatial.distance.cdist(emb1, emb2, DISTANCE_METRIC).item()


def enroll_new_user(username):
    fpath = os.path.join(ENROLLMENT_FOLDER, username + '_' + ENROLL_RECORDING_FNAME)
    emb = get_emb(fpath)
    store_user_embedding(username, emb)


def verify_user(username):
    fpath = os.path.join(VERIFICATION_FOLDER, username + '_' + VERIFY_RECORDING_FNAME)
    #username + '_' + VERIFY_RECORDING_FNAME
    emb = get_emb(fpath)
    speaker_models = load_speaker_models()
    dist = emb_dist(emb, speaker_models[username])
    print(dist)
    return dist > THRESHOLD, fpath


def identify_user():
    fpath = os.path.join(VERIFICATION_FOLDER, IDENTIFY_RECORDING_FNAME)
    emb = get_emb(fpath)
    speaker_models = load_speaker_models()
    dist = [(other_user, emb_dist(emb, speaker_models[other_user]))
            for other_user in speaker_models]
    print(dist)
    username, min_dist = min(dist, key=lambda x:x[1])

    if min_dist > THRESHOLD:
        return username, fpath
    return None, fpath


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
        users_list = show_current_users()
        if not users_list:
            print("No users found")
        else:
            print("\n".join(users_list))

    elif args.enroll:
        username = args.username
        assert username is not None, "Enter username"
        assert username not in show_current_users(), "Username already exists in database"
        enroll_new_user(username)

    elif args.verify:
        username = args.username
        assert username is not None, "Enter username"
        assert username in show_current_users(), "Unrecognized username"
        verified, fpath = verify_user(username)
        if verified:
            print("User verified")
        else:
            print("Unknown user")
        var = input("Save recording: (y/n)?")
        if var == 'y' or var == 'yes':
            print(f'{fpath} saved')
        elif var == 'n' or var == 'no':
            os.remove(fpath)
            print(f'{fpath} removed')
        

    elif args.identify:
        identified_user, fpath = identify_user()
        print("Identified User {}".format(identified_user))
        var = input("Save recording? (y/n):")
        if var == 'y' or var == 'yes':
            print(f'{fpath} saved')
        elif var == 'n' or var == 'no':
            os.remove(fpath)
            print(f'{fpath} removed')

    elif args.delete:
        username = args.username
        assert username is not None, "Enter username"
        assert username in show_current_users(), "Unrecognized username"
        delete_user(username)

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

