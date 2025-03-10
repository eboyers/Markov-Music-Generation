import os
import mido
import random

from mido import MidiFile, MidiTrack, Message

class MIDIProcessor:
    def __init__(self):
        """
        Class to extract note sequences from MIDI files
        """
        self.midi_dir = "classical_midis"
        self.composers = self._get_composers()
        
    def _get_composers(self):
        """Get list of composers from directory structure"""
        return [d for d in os.listdir(self.midi_dir) if os.path.isdir(os.path.join(self.midi_dir, d))]
    
    def extract_notes(self, midi_file_path):
        """
        Extract a sequence of notes from a MIDI file
        """
        notes = []
        mid = MidiFile(midi_file_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
        return notes
    
    def get_composer_note_sequences(self, composer):
        """
        Get note sequences for all MIDI files of a composer
        """
        composer_dir = os.path.join(self.midi_dir, composer)
        note_sequences = {}
        
        for file in os.listdir(composer_dir):
            filepath = os.path.join(composer_dir, file)
            notes = self.extract_notes(filepath)
            note_sequences[file] = notes
            
        return note_sequences
    
    def get_all_note_sequences(self):
        """
        Get note sequences for all composers
        """
        all_sequences = {}
        for composer in self.composers:
            all_sequences[composer] = self.get_composer_note_sequences(composer)
        return all_sequences

    def create_midi(self, notes, durations, output_file, tempo=500000):
        """
        Create a minimal MIDI file from note/duration pairs, with no extras.
        """
        # create new MIDI file and track
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # set instrument to piano
        track.append(Message('program_change', program=0, time=0))
        
        # set tempo meta-message
        tempo_msg = mido.MetaMessage('set_tempo', tempo=tempo)
        track.append(tempo_msg)
        
        current_time = 0
        timeline = []
        
        # build timeline of note_on and note_off events
        for note, duration in zip(notes, durations):
            if duration < 0:
                # negative duration is a rest: just advance the time
                current_time += abs(duration)
            else:
                # schedule note_on at the current_time
                timeline.append((current_time, 'note_on', note, 64))
                
                # schedule note_off after 'duration' ticks
                end_time = current_time + duration
                timeline.append((end_time, 'note_off', note, 0))
                
                # advance current_time by this duration
                current_time += duration
        
        # sort events by time
        timeline.sort(key=lambda x: x[0])
        
        last_time = 0
        # convert timeline into MIDI messages
        for event in timeline:
            event_time, event_type, note, velocity = event
            delta = event_time - last_time  # time since last event
            
            if event_type == 'note_on':
                track.append(Message('note_on', note=note, velocity=velocity, time=delta))
            elif event_type == 'note_off':
                track.append(Message('note_off', note=note, velocity=velocity, time=delta))
            
            last_time = event_time
        
        mid.save(output_file)

