import streamlit as st
import os
"""
# Membrane 2.0
Rapid Voice-Based Biometric Authentication
"""
# os.system('python VBBA.py -v -u jun')
import webbrowser
from network import *

from membrane_st import *
import time
from pydub import AudioSegment
import io
# if st.button('Say hello'):
#   st.write('Why hello there')
# else:
#   st.write('Goodbye')
# if st.checkbox('Enroll'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])
option = st.selectbox(
    'Please select a task.',
     ('Please select (Enroll, Verify)','Enroll', 'Verify'))

url = 'https://www.streamlit.io/'

# def do_enroll(username, audio_file, phrase):
def fpath_last_number(fpath, extension = '.wav'):
    prev = fpath
    while os.path.exists(fpath+extension):
        prev = fpath
        if fpath[-1].isalpha():
            fpath = fpath+'2'
        else:
            fpath = fpath[:-1]+str(int(fpath[-1])+1)
    return prev

def correct_file(written_username, correction, extension = '.wav'):
    FOLDER = VERIFICATION_FOLDER
    fpath = os.path.join(FOLDER, written_username)
    fpath = fpath_last_number(fpath)
    new_fpath = os.path.join(FOLDER, correction)
    new_fpath = fpath_numbering(new_fpath)
    os.rename(fpath+extension, new_fpath+extension)

def save_file(denoised_data, username, enroll = False):
    # st.write('\n')
    # save1 = st.text_input('Can we save ? (y/n):')#processed data of your audio file in our database for testing and model development

    # save = st.selectbox(
    # 'Please select yes/no',
    #  ('Please select (yes/no))','yes', 'no'))
    # if save =='y' or save =='yes':
    if enroll: FOLDER = ENROLLMENT_FOLDER
    else: FOLDER = VERIFICATION_FOLDER
    fpath = os.path.join(FOLDER, username)
    fpath = fpath_numbering(fpath)
    write_recording(fpath,  denoised_data)
    st.write('File saved. Thank you!')
    # leave = True
    # elif save == 'n' or save == 'no':
        # st.write('Thank you!')
        # leave = True
        # st.write(save)
    # else:
        # leave = False
    # st.write('Thank you!')
    # if leave:
    # st.write('Thank you so much for your testing!')
    st.write('\n')


def save_comments():
    comments = st.text_input('Please write here any comments you have. Thank you!:')
    if comments:
        comments_path = os.path.join(COMMENTS_FOLDER, 'comments.txt')
        with open(comments_path,'a') as f:
            f.write(comments)
            f.write('\n\n')
        st.write('Comments saved! Thank you!')

def ask_save_file():
    save = st.selectbox(
        'Can we save processed data of your audio file in our database for testing and model development? (y/n)'
        ,('Please select (Yes/No)','Yes','No'))
    # if save =='yes':
        # save_file(denoised_data, username)
        # st.write('Thank you so much!')
    # elif save == 'no':
        # os.remove()
        # st.write('Thank you for testing!')
    return save.lower()
# work_done = False
# leave = False
def audio_encoding(audio_file):
    option = st.selectbox(
    'Please select a format for the proper encoding.',
     ('Please select (wav, mp3, m4a, ...)','wav', 'mp3','m4a','flac','other')
    )
    ext = option
    if option =='wav':
        return audio_file
    elif option =='other':
        ext = st.text_input('Please enter the format:')
    if 'Please' not in ext and ext!='other':
        audio_file = AudioSegment.from_file(audio_file, format=ext)
        audio_file.export('tested_users/audio_test'+'.wav', format="wav")
        # st.write(len(np.frombuffer(audio_file)))
        # data, sr = sf.read(audio_file)
        # st.write( type(audio_file))
        # st.write( type(audio_file.read()))
        # song = AudioSegment.from_file(audio_file, format="mp3")
        # st.write( len(np.frombuffer(audio_file.getbuffer(),dtype=np.int16)))
        # sf.read(AudioSegment.from_mp3(audio_file) )#, format="mp3"
        audio_file = 'tested_users/audio_test'+'.wav'
        return audio_file



def iphone_note():
    st.text(
    '''
    Note for iphone users: you can upload your recordings from Voice Memos by
    1. Open the Voice Memos app on your iphone.
    2. Tap on the desired memo.
    3. Tap on the "Share" icon.
    4. Choose the destination folder.
    5. You can find your memo you just copied when "browse files".
    ''')


if option == 'Enroll':

    username = st.text_input('Enter your username:')
    st.text('Your username can be public in future hacker-mode testing. You can use any nickname.')
    if username == '':
        st.write('Please enter your username.')
    else:
        if username in show_current_users():
            var = st.text_input('Username already exists in database. Do you want to replace? (y/n):')
            if var == 'y' or var =='yes':
                phrase = st.text_input('Enter your secret phrase (leave blank for auto detection):')
                audio_file = st.file_uploader('Your speaking audio file:')
                iphone_note()
                audio_file = audio_encoding(audio_file)
                save = ask_save_file()
                # denoised_data = enroll_new_user(username, file = audio_file, phrase = phrase)
                if st.button('Start'):
                    denoised_data = enroll_new_user(username, file = audio_file, phrase = phrase)
                    st.write(f'User - {username} - enrolled!')
                    # st.write()
                    # st.write('Can we save processed data of your audio file in our database for testing and model development? (y/n)')
                    # if st.button('y'):
                    if save == 'yes':
                        save_file(denoised_data, username, enroll = True)
                    # if st.button('n'):
                    #     st.write('nn Thank you!')
                else:
                    st.write('Please click the Start button to enroll. It will take a few seconds.')
                # work_done = True
            # username = st.text_input("Username: ")
            # st.write(username)
            # time.sleep(5)
            # st.write('this?')
            # time.sleep(5)
            # do_enroll(username)
                # else:
                #     'Please click the Start button to enroll. It will take a few seconds.'
            elif var =='n' or var =='no':
                st.write('Please select other tasks.')
                # work_done = True
                # work_done = True #########

        else:
            phrase = st.text_input('Enter your secret phrase (leave blank for auto detection):')
            audio_file = st.file_uploader('Your speaking audio file:')
            iphone_note()
            audio_file = audio_encoding(audio_file)
            save = ask_save_file()
            if st.button('Start'):
                denoised_data = enroll_new_user(username, file = audio_file, phrase = phrase)
                st.write(f'User - {username} - enrolled!')
                if save == 'yes':
                    save_file(denoised_data, username, enroll = True)
                # work_done()


elif option == 'Verify':
    audio_file = st.file_uploader('Your speaking audio file:')
    iphone_note()
    audio_file = audio_encoding(audio_file)

    # audio_file = AudioSegment.from_file(audio_file, format="m4a")
    # audio_file.export('tested_users/audio_test'+'.wav', format="wav")
    # # st.write(len(np.frombuffer(audio_file)))
    # # data, sr = sf.read(audio_file)
    # # st.write( type(audio_file))
    # # st.write( type(audio_file.read()))
    # # song = AudioSegment.from_file(audio_file, format="mp3")
    # # st.write( len(np.frombuffer(audio_file.getbuffer(),dtype=np.int16)))
    # # sf.read(AudioSegment.from_mp3(audio_file) )#, format="mp3"
    # audio_file = 'tested_users/audio_test'+'.wav'
    save = ask_save_file()
    if st.button('Start'):
        verified,  username, denoised_data = verify_user(file = audio_file)
        if verified:
            st.write('Congratulation!')
            st.write(f'User - {username} - verified.\n')
            # st.write('If not correct please provide your username:')
            # correction = st.text_input('Correct username (leave blank if it was correct):')
            # if correction:
            #     correct_file(username , correction)
            if save == 'yes':
                save_file(denoised_data, username)
        else:
            st.write(f'Unknown user.')
            # correction = st.text_input('Correct username:')
            # if correction:
            #     correct_file(username , correction)
        # work_done()
            if save == 'yes':
                save_file(denoised_data, 'unknown')
    else:
        'Please click the Start button to verify. It will take a few seconds.'

# st.write('\n')
st.write(' ')
if option == 'Verify' and save == 'yes':
    st.write('If the verified user is not correct, please let us know your correct username.')
    veri_name = st.text_input('Verified name: (for "Unkown user", please leave blank):')
    correct_name = st.text_input('Your correct name:')
    if correct_name:
        if veri_name =='': veri_name = 'unknown'
        correct_file(veri_name, correct_name)
st.write(' ')
# st.write('\n')
save_comments()
st.write(' ')
st.write(' ')
'''

'''
st.write('Please refresh the page for a new task.')
st.write(' ')
st.write('Jun Seok Lee')
st.write('lee.junseok39@gmail.com')
st.write('Collaborating with OMIC.ai')
# save = st.text_input('Enter we save file?::::')
# save_file(denoised_data, username, save)
# save = st.text_input('Enter we save file?:')
# # save = st.selectbox(
# # 'Please select yes/no',
# #  ('Please select (yes/no))','yes', 'no'))
# if save =='y' or save =='yes':
#     fpath = os.path.join(VERIFICATION_FOLDER, username)
#     fpath = fpath_numbering(fpath)
#     write_recording(fpath,  denoised_data)
#     st.write('File saved. Thank you!')
#     leave = True
#     st.write(save)
#     st.button('haha')
# elif save == 'n' or save == 'no':
#     st.write('Thank you!')
#     leave = True
#     st.write(save)
# else:
#     leave = False
# st.write('Thank you!')
# if leave:
#     st.write('Thank you so much for your testing!')
#     st.write('\n')
#     comments = st.text_input('Please write here any comments you have. Thank you!:')
#     comments_path = os.path.join(COMMENTS_FOLDER, 'comments.txt')
#     with open(comments_path,'a') as f:
#         f.write(comments)
#         f.write('\n')


  # username = st.text_input("Username: ")
  # username
# genre = st.radio(
#     "What's your favorite movie genre",
#     ('Comedy', 'Drama', 'Documentary'))
# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn't select comedy.")
