import pyaudio
import wave


def list_device():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i).get('name'))


# USB mic should be index 1
def record_audio(time:float=1, name:str="temp.wav"):
    # audio setting
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = time # seconds to record
    dev_index = 1 # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = name # name of .wav file

    audio = pyaudio.PyAudio()

    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
        input_device_index = dev_index,input = True, \
        frames_per_buffer=chunk)
    
    print("recording")
    frames = []

    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)

    print("finished recording")
 
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


def get_audio_arr_old(filename:str="./audio/jp.wav"):
    ifile = wave.open(filename)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    return audio_as_np_int16

