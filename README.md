# membrane

Rapid voice-based biometric authentication.

We have a web-based biomedical/AI platform used for open-source COVID-19 research. Users currently log in via magic link, but we would like to add an option where the person is identified via voice through the browser. Ideally we would generate a phrase for the user to speak into a laptop/PC microphone, at which time the user would be rapidly authenticated and would be granted access.

**Contributor**: [Jun Seok Lee](lee.junseok39@gmail.com).

## TODO

- [ ] Jun registers for [OS](os.omic.ai).
- [x] Jun signs Nondisclosure Agreement.
- [ ] Profiling visualization dataset created -- 200 samples.
- [ ] Training visualization dataset created -- 1K+ samples.

## Background

TODO

## Data

There are a plethora of voice identification and transcription datasets publicly available, including FSDD, VoxCeleb, CommonVoice, LibreSpeech, etc.  These existing datasets will require transfer learning and need to (optimistically) go through various real-world transformations to match production environments, which primarily include non-linear speech cadence/stuttering/filler words and static/dynamic environment background noise.  In total, the datasets accumulate to over 2TB.

### Structure

TODO

### Privacy

The data provided for the development of this model is highly protected; access to this data is permissable only by the signing of our Nondisclosure Agreement and under no condition can be distributed outside of Omic, Inc.

Software developed on top of this data, however, are openly shareable, so long as they do not immediately pose a risk of the above data policy.

*tl;dr don't share data and be mindful with sharing how you interface with the data.*

## Model

Weights location:  https://membrane.s3-us-west-2.amazonaws.com/checkpoint_20200604-112741_0.00098080572206527.pth.tar.

We would like a voiceprint authentication model that isolates the user's voice based on the prescribed phrase and then authenticates the user based on a voice match to the training phrase.

As an example, we could provide several training phrases for the user to speak in our application through the laptop microphone. We would record these phrases, then provide a new login phrase for them to read and log in. The person’s voice would need to be isolated from background noise. 

The model would score the new login phrase - identifying the person of closest match and false positive/negative (precision/recall) data for understanding false positive and negative rates. The goal would be for these to score at least as well as common voice recognition models that currently exist.

### Tier 1

TODO

### Tier 2

TODO

### Tier 3

TODO

### Tier 4 (Boss Level)

Collect and package all references and results.  Publish a paper in a top journal.  Save the princess.
