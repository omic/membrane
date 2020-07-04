############## READING and WRITING functions ##########################
########## Let me use my pre-defined functions below
########## Feel free to amend!
#
#
# # TODO:  Start storing this data in S3, not the EC2.
# INDEX_PATH = '~/membrane/users/index.json'
# SAMPLE_PATH = '~/membrane/users/samples'

# def _read_db(email: str) -> dict:
#     """Get user phrase and location to all samples files pertaining
#     to this user.
#     """
#     try:
#         with open(INDEX_PATH, 'rt') as user_index:
#             idx = json.load(user_index)
#             entry = idx[email]
#             # TODO:  Return filenames and secret phrase for user with this
#             #        email
#     except:
#         return None

# def _write_db(email: str, phrase: str = None, sample: bytes = None) -> None:
#     """Write either phrase or new sample to user entry in index."""
#     if sample:
#         # TODO:  Write to SAMPLE_PATH and get path to file.
#         sample_path = '...'
#     with open(INDEX_PATH, 'rt+') as user_index:
#         idx = json.load(user_index)
#         entry = {}
#         if sample_path:
#             # TODO:  Update sample path to entry.
#             pass
#         if phrase:
#             # TODO:  Add phrase to entry.
#             pass
#         json.dump(idx, user_index)
############