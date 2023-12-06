## beat.Gen by v @ violet castles . com <3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import librosa
import numpy as np
from stftpitchshift import StftPitchShift
import random
import os
import tkinter as tk
from tkinter import filedialog
from pedalboard import *
import pygame.mixer
import time

song_exported = False

def create_ui():
       # Add player controls
    def play_song():
        pygame.mixer.music.play()  # Play the song

    def stop_song():
        pygame.mixer.music.stop()  # Stop the song
    window = tk.Tk()
    window.title("break.Gen")

    tk.Label(window, text="Song path:").grid(row=0)
    song_path_entry = tk.Entry(window)
    song_path_entry.grid(row=0, column=1)

    tk.Label(window, text="Desired BPM:").grid(row=1)
    bpm_entry = tk.Scale(window, from_=180, to=250, orient=tk.HORIZONTAL)
    bpm_entry.grid(row=1, column=1)

    tk.Label(window, text="Chop Intensity:").grid(row=2)
    intensity_entry = tk.Scale(window, from_=1, to=9, orient=tk.HORIZONTAL)
    intensity_entry.grid(row=2, column=1)

    tk.Label(window, text="Hyperpop vocals:").grid(row=3)
    hyperpop_var = tk.StringVar(value="n")
    hyperpop_entry = tk.Checkbutton(window, variable=hyperpop_var, onvalue="y", offvalue="n")
    hyperpop_entry.grid(row=3, column=1)
    
    """
    tk.Label(window, text="Reverb vocals:").grid(row=4)
    reverb_vox_var = tk.StringVar(value="n")
    reverb_vox_entry = tk.Checkbutton(window, variable=reverb_vox_var, onvalue="y", offvalue="n")
    reverb_vox_entry.grid(row=4, column=1)

    tk.Label(window, text="Distort backing:").grid(row=5)
    distort_backing_var = tk.StringVar(value="n")
    distort_backing_entry = tk.Checkbutton(window, variable=distort_backing_var, onvalue="y", offvalue="n")
    distort_backing_entry.grid(row=5, column=1)

    tk.Label(window, text="Reverb backing:").grid(row=6)
    reverb_backing_var = tk.StringVar(value="n")
    reverb_backing_entry = tk.Checkbutton(window, variable=reverb_backing_var, onvalue="y", offvalue="n")
    reverb_backing_entry.grid(row=6, column=1)

    tk.Label(window, text="Random break effects:").grid(row=7)
    break_fx_var = tk.StringVar(value="n")
    break_fx_entry = tk.Checkbutton(window, variable=break_fx_var, onvalue="y", offvalue="n")
    break_fx_entry.grid(row=7, column=1)

    """

    def start_process():
        separator = Separator('spleeter:5stems')
        audio_loader = AudioAdapter.default()
        pygame.mixer.init()
        global hyperpop, reverb_vox, intensity, song_path, bpm, distort_backing, reverb_backing, break_fx
        hyperpop = hyperpop_var.get()
        ##reverb_vox = reverb_vox_var.get()
        ##distort_backing = distort_backing_var.get()
        ##reverb_backing = reverb_backing_var.get()
        ##break_fx = break_fx_var.get()
        intensity = intensity_entry.get()
        song_path = song_path_entry.get()
        bpm = bpm_entry.get()

        main()

        while not song_exported:
            time.sleep(1)  # Wait for 1 second

        start_button.grid_remove()

        new_export = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 files", "*.mp3")])
        audio_loader.save(new_export, final, separator._sample_rate, "mp3", "128k")
        pygame.mixer.music.load(export_path) #load the export

        play_button = tk.Button(window, text='Play', command=play_song)
        stop_button = tk.Button(window, text='Stop', command=stop_song)

        def exit():
            pygame.mixer.music.stop()
            window.quit()
        
        def restart_process():
            global song_exported
            song_exported = False
            start_process()
        
        play_button.grid(row=9, column=1)
        stop_button.grid(row=10, column=1)
        exit_button = tk.Button(window, text='Exit', command=exit)
        exit_button.grid(row=11, column=1)
        restart_button = tk.Button(window, text="Restart", command=restart_process)
        restart_button.grid(row=12, column=1)
    
    start_button = tk.Button(window, text="Start", command=start_process)
    start_button.grid(row=8, column=1)

    window.mainloop()

def timefix(breakbeat):
    # Normalize the breakbeat
    breakbeat = normalize_audio(breakbeat)
    # Track the beats in the breakbeat
    _, beat_frames = librosa.beat.beat_track(breakbeat, sr=sr)

    # If the breakbeat is longer than 4 bars, keep only the first 4 bars
    if len(beat_frames) > 16:
        beat_frames = beat_frames[:16]
        breakbeat = breakbeat[:beat_frames[-1]]
    # If the breakbeat is shorter than 4 bars, repeat sections until it's 16 beats long
    elif len(beat_frames) < 16:
        repeats = (16 // len(beat_frames)) + 1  # Calculate the number of repeats needed
        breakbeat = np.tile(breakbeat, repeats)  # Repeat the breakbeat
        breakbeat = breakbeat[:(16 * sr)]  # Trim the repeated breakbeat to 16 beats

    rms = np.sqrt(np.mean(breakbeat**2))
    desired_rms = 10**(-3/20)  # Convert -3dB to linear scale
    gain = desired_rms / rms
    breakbeat = breakbeat * gain


    return breakbeat

def replace_drums_with_breakbeats(bars, breakbeats, sr, energy_scores, intensity, drums):
    new_bars = []
    
    # Calculate energy thresholds based on percentiles
    high_energy_threshold = np.percentile(energy_scores, 70)
    medium_energy_threshold = np.percentile(energy_scores, 35)

    for i in range(0, len(bars), 4):  # Step through bars 4 at a time
        avg_energy = np.mean(energy_scores[i:i+4])

        if avg_energy > high_energy_threshold:
            breakbeat = random.choice(breakbeats['high'])
            energy = 'high'
        elif avg_energy > medium_energy_threshold:
            breakbeat = random.choice(breakbeats['medium'])
            energy = 'medium'
        else:
            breakbeat = random.choice(breakbeats['low'])
            energy = 'low'

        breakbeat=timefix(breakbeat)


        # Sample chopping based on intensity
        chop_chance = {1: 0.5, 2: 0.5, 3: 0.8, 4: 0.6, 5: 0.6, 6: 0.9, 7: 0.6, 8: 0.9, 9: 0.5}
        chop_amount = {1: 2, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 8, 8: 8, 9: 16}
        else_amount = {1: 4, 2: 4, 3: 4, 4: 2, 5: 2, 6: 2, 7: 4, 8: 4, 9: 8}

        """if random.random() < .3:
            if random.random() < chop_chance[intensity]:
                new_extend = chop_and_replace(breakbeat, chop_amount[intensity], breakbeats, energy)
                effected = np.array([effect(new_extend)])
                new_bars.extend(effected)
            else:
                new_extend = chop_and_replace(breakbeat, else_amount[intensity], breakbeats, energy)
                effected = np.array([effect(new_extend)])
                new_bars.extend(effected)
        else: """
        if random.random() < chop_chance[intensity]:  
            new_bars.extend(chop_and_replace(breakbeat, chop_amount[intensity], breakbeats, energy))
        else:
            new_bars.extend(chop_and_replace(breakbeat, else_amount[intensity], breakbeats, energy))
        

    new_bars = np.concatenate(new_bars)[:len(drums)]
    return new_bars

def normalize_audio(audio):
    return librosa.util.normalize(audio)

def Distort_fx(audio):
    audio = audio.astype(np.float32)
    audio = np.array(audio).flatten()
    board = Pedalboard([Distortion()])
    audio = board.process(audio, sr)
    return audio

def Reverb_fx(audio):
    audio = audio.astype(np.float32)
    audio = np.array(audio).flatten()
    board = Pedalboard([Reverb()])
    audio = board.process(audio, sr)
    return audio

def effect(audio):

    """ effects = ['phaser', 'reverb', 'delay', 'chorus', 'distortion', 'high_pass', 'low_pass', 'bitcrush']

    # Choose a random effect
    effect = random.choice(effects)
    if break_fx.lower() == "y":
        audio = audio.astype(np.float32)
        audio = np.array(audio).flatten()
        # Apply the chosen effect
        if effect == 'phaser':
            board = Pedalboard([Phaser()])
            audio = board.process(audio, sr)
        elif effect == 'reverb':
            board = Pedalboard([Reverb()])
            audio = board.process(audio, sr)
            pass
        elif effect == 'delay':
            board = Pedalboard([Delay()])
            audio = board.process(audio, sr)
            pass
        elif effect == 'chorus':
            board = Pedalboard([Chorus()])
            audio = board.process(audio, sr)
            pass
        elif effect == 'distortion':
            board = Pedalboard([Distortion()])
            audio = board.process(audio, sr)
            pass
        elif effect == 'high_pass':
            board = Pedalboard([HighpassFilter()])
            audio = board.process(audio, sr)
        elif effect == 'low_pass':
            board = Pedalboard([LowpassFilter()])
            audio = board.process(audio, sr)
        elif effect == 'bitcrush':
            board = Pedalboard([Bitcrush()])
            audio = board.process(audio, sr)
        return audio
    
    else: """
    return audio

def chop_and_replace(breakbeat, n, breakbeats, energy_level):
    to_chop = np.array_split(breakbeat, n)
    chopped = []
    
    last_two_replacements = [None, None]  # Keep track of the last two replacements

    for i in range(len(to_chop)):
        while True:  # Keep trying until we get a new replacement
            replacement = np.array(random.choice(breakbeats[energy_level]))  # Convert to numpy array
            if replacement.tolist() not in last_two_replacements:  # If the replacement is new, break the loop
                break

        replacement = timefix(replacement)
        segment_length = len(to_chop[i])
        start = int((i / n) * segment_length)
        end = start + segment_length
        to_chop[i] = replacement[start:end]
        # Update the last two replacements
        last_two_replacements[0] = last_two_replacements[1]
        last_two_replacements[1] = replacement.tolist()
        chopped.append(to_chop[i])
    return chopped

def main():
    global hyperpop, intensity, song_path, bpm
    separator = Separator('spleeter:5stems')

    # Using custom configuration file.
    audio_loader = AudioAdapter.default()
    sample_rate = 44100
    global sr
    sr = sample_rate
    waveform, _ = audio_loader.load(song_path, sample_rate=sample_rate)
    prediction = separator.separate(waveform)

    # Now add up all that is not drums
    vocals = prediction["vocals"]
    backing = prediction["bass"]
    for key in ["other"]:
        backing += prediction[key]
    
    drums = prediction["drums"]
    breakbeats = {}
    sample_rate = 44100  # or whatever your sample rate is
    sample_energy = ['high', 'medium', 'low']

    for energy in sample_energy:
        breakbeats[energy] = [librosa.load(os.path.join("breaks", energy, f), sr=sample_rate)[0] for f in os.listdir(os.path.join("breaks", energy)) if f.endswith('.wav')]
    
    vocals = np.array(vocals, dtype=np.float32)
    backing = np.array(backing, dtype=np.float32)
    drums = np.array(drums, dtype=np.float32)

    drums = np.mean(drums, axis=1) # convert to mono

    print(drums);

    tempo, beat_frames = librosa.beat.beat_track(y=drums, sr=sample_rate)

    ## determine if song is in 3 or 4, or other time

    bars = len(beat_frames) // 4
    bars = np.array_split(drums, bars)

    energy_scores = [np.sum(np.abs(bar)) for bar in bars]

    avg_energy = np.average(energy_scores)

    print (f"Average energy: {avg_energy}");

    tempo, beat_frames = librosa.beat.beat_track(drums, sr=sample_rate)
    
    if tempo < 80: #this allows us to make sure the song is properly sped up
        tempo *= 2

    print(f"Estimated tempo: {tempo} beats per minute")

    # postpunk avg energy = 10364
    # cyan hardcore avg energy = 12600
    # lsd avg energy = 2000

    high_energy_threshold = 110000
    medium_energy_threshold = 50000

    bpm = int(bpm) # convert bpm into an int

    if bpm < 180:
        print("Breakcore must be at least 180BPM according to the Geneva Convention, setting to 230BPM")
        bpm=230

    speed_factor = bpm / tempo

    vocals = np.mean(vocals, axis=1)
    backing = np.mean(backing, axis=1)

    drums = librosa.effects.time_stretch(drums, speed_factor)
    vocals = librosa.effects.time_stretch(vocals, speed_factor)
    backing = librosa.effects.time_stretch(backing, speed_factor)


    if hyperpop.lower() == "y":
        pitchshifter = StftPitchShift(1024, 256, sample_rate)
        vocals = pitchshifter.shiftpitch(vocals, 1.5, quefrency=0.0) #formant shifter

    bars = replace_drums_with_breakbeats(bars, breakbeats, sample_rate, energy_scores, intensity, drums)
    
    print(len(bars))
    print(len(vocals))
    print(len(backing))

    def compress(audio, threshold=-3, ratio=3): #basic compressor implementation
        # Convert threshold from dB to linear
        threshold_linear = 10.0**(threshold / 20)

        # Calculate gain
        gain = np.where(audio > threshold_linear, ratio * (audio - threshold_linear), 0)

        # Apply gain to audio
        compressed_audio = audio - gain

        return compressed_audio

    def apply_gain(audio, target_db=-3):
        # Convert target dB to linear
        target_linear = 10.0**(target_db / 20)

        # Calculate current level
        current_level = np.sqrt(np.mean(audio**2))

        # Calculate required gain
        gain = target_linear / current_level

        # Apply gain to audio
        audio = audio * gain

        return audio

    def agc(audio, target_rms=None): #automated gain control
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio**2))

        # If no target RMS is provided, set it to the current RMS
        if target_rms is None:
            target_rms = current_rms

        # Calculate required gain
        gain = target_rms / current_rms

        # Apply gain to audio
        agc_audio = audio * gain

        return agc_audio
    


    # Normalize each track
    #vocals = normalize_audio(vocals)
    #backing = normalize_audio(backing)
    #bars = normalize_audio(bars)

    # Calculate the RMS of each track
    vocals_rms = np.sqrt(np.mean(vocals**2))
    backing_rms = np.sqrt(np.mean(backing**2))
    drums_rms = np.sqrt(np.mean(drums**2))

    # Find the minimum RMS
    min_rms = min(vocals_rms, backing_rms, drums_rms)

    # Use the minimum RMS as the target for the AGC
    vocals = agc(vocals, min_rms)
    backing = agc(backing, min_rms)
    drums = agc(drums, min_rms)

    # Compress each track - maybe remove this one
    vocals = compress(vocals)
    backing = compress(backing)
    drums = compress(drums)

    # Apply gain to bring each track up to -3 dB
    vocals = apply_gain(vocals)
    backing = apply_gain(backing)
    drums = apply_gain(drums)



    """
    if distort_backing.lower() == "y":
        backing = Distort_fx(backing)

    if reverb_backing.lower() == "y":
        backing = Reverb_fx(backing)

    if reverb_vox.lower() == "y":
        vocals = Reverb_fx(vocals)
    """

    vocals = vocals.reshape(-1, 1)
    backing = backing.reshape(-1, 1)
    bars = bars.reshape(-1, 1)

    global final
    final = (vocals + backing + bars) / 3
    final /= np.max(np.abs(final))
    final = final.reshape(-1, 1)
    def export():
        global export_path
        export_path = "output/export_file.mp3"   
        audio_loader.save("output/export_file.mp3", final, separator._sample_rate, "mp3", "128k")
        global song_exported
        song_exported = True
    export()
if __name__ == '__main__': 
    create_ui()