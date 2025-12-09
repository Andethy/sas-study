import { useEffect, useRef, useState } from 'react';

interface MidiControllerState {
  k1Knob: number;  // Joystick Y-axis (0-1, bottom to top)
  k2Knob: number;  // Timbre volume (0-1)
  k3Knob: number;  // Timbre mix (0-1)
  k4Knob: number;  // Master volume (0-1)
  joystickX: number;  // Joystick X (-1 to 1)
  joystickY: number;  // Joystick Y (0-1, bottom to top)
}

interface UseMidiInputProps {
  onTimbreVolumeChange?: (volume: number) => void;
  onTimbreMixChange?: (mix: number) => void;
  onMasterVolumeChange?: (volume: number) => void;
  onJoystickChange?: (x: number, y: number) => void;
  onDrumPadPress?: (padNumber: number, velocity: number) => void;
  onKeyboardNote?: (note: number, velocity: number, isNoteOn: boolean) => void;
  onLog?: (message: string, type: 'info' | 'success' | 'error') => void;
}

export const useMidiInput = ({
  onTimbreVolumeChange,
  onTimbreMixChange,
  onMasterVolumeChange,
  onJoystickChange,
  onDrumPadPress,
  onKeyboardNote,
  onLog
}: UseMidiInputProps) => {
  const [isConnected, setIsConnected] = useState(false);
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const [controllerState, setControllerState] = useState<MidiControllerState>({
    k1Knob: 0,    // Joystick Y (bottom)
    k2Knob: 0.8,  // Default timbre volume
    k3Knob: 0.5,  // Default timbre mix
    k4Knob: 0.8,  // Default master volume
    joystickX: 0,
    joystickY: 0  // Bottom position
  });

  const midiInputRef = useRef<WebMidi.MIDIInput | null>(null);
  const lastUpdateTimeRef = useRef<{ [key: string]: number }>({});

  // Helper function to convert MIDI note to note name
  const getNoteNameFromMidi = (midiNote: number): string => {
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(midiNote / 12) - 1;
    const noteName = noteNames[midiNote % 12];
    return `${noteName}${octave}`;
  };

  // AKAI MPK Mini MIDI CC mappings (shifted to accommodate joystick Y)
  const MIDI_CC = {
    K1_KNOB: 1,    // K1 knob (Modulation) -> Joystick Y-axis (same as joystick Y)
    K2_KNOB: 2,    // K2 knob (Breath Controller) -> Timbre Volume
    K3_KNOB: 3,    // K3 knob -> Timbre Mix
    K4_KNOB: 4,    // K4 knob -> Master Volume
    JOYSTICK_Y: 1  // Joystick Y sends CC1 (same as K1)
  };
  
  // Pitch bend for joystick X (different message type)
  const PITCH_BEND_STATUS = 224; // 0xE0
  
  // MIDI Note On/Off for drum pads
  const NOTE_ON_STATUS = 144; // 0x90
  const NOTE_OFF_STATUS = 128; // 0x80
  
  // AKAI MPK Mini drum pad note mappings (common mappings to try)
  const DRUM_PAD_NOTES = {
    // Standard GM drum mapping
    36: 'snare',  // Kick/Pad 1 -> Snare fills
    37: 'toms',   // Stick/Pad 2 -> Tom fills  
    38: 'hats',   // Snare/Pad 3 -> Hi-hat fills
    // Alternative AKAI mappings
    40: 'snare',  // Alternative snare
    41: 'toms',   // Alternative toms
    42: 'hats',   // Alternative hats
    // Chromatic pad mappings (C3-B3)
    48: 'snare',  // C3
    49: 'toms',   // C#3
    50: 'hats',   // D3
    // Higher octave (C4-B4)  
    60: 'snare',  // C4
    61: 'toms',   // C#4
    62: 'hats'    // D4
  };

  const throttle = (key: string, callback: () => void, delay: number = 50) => {
    const now = Date.now();
    const lastUpdate = lastUpdateTimeRef.current[key] || 0;
    
    if (now - lastUpdate >= delay) {
      callback();
      lastUpdateTimeRef.current[key] = now;
    }
  };

  const handleMidiMessage = (event: WebMidi.MIDIMessageEvent) => {
    const [status, dataOrCC, value] = event.data;
    
    // Debug: Log all MIDI messages to help identify notes and controls
    console.log('MIDI Message:', { 
      status, 
      dataOrCC, 
      value, 
      hex: `0x${status.toString(16)}`,
      type: status === 144 ? 'Note On' : status === 128 ? 'Note Off' : status === 176 ? 'CC' : status === 224 ? 'Pitch Bend' : 'Other'
    });
    
    // Handle Note On messages (status 144 = 0x90)
    if (status === NOTE_ON_STATUS) {
      const note = dataOrCC;
      const velocity = value;
      
      console.log('🎵 Note On detected:', { note, velocity, fillType: DRUM_PAD_NOTES[note] });
      
      // Check if it's a drum pad first (prioritize drum pads)
      if (DRUM_PAD_NOTES[note] && velocity > 0) {
        const padNumber = Object.keys(DRUM_PAD_NOTES).indexOf(note.toString()) + 1;
        console.log('🥁 Drum pad triggered:', { padNumber, note, velocity });
        onDrumPadPress?.(padNumber, velocity / 127); // Normalize velocity to 0-1
        return; // Don't process as keyboard note
      } 
      
      // Fallback drum pad detection for unmapped notes in drum range
      if (velocity > 0 && note >= 36 && note <= 51) {
        console.log('🥁 Unmapped drum pad note:', note);
        const padNumber = ((note - 36) % 3) + 1; // Cycle through pads 1-3
        const fillTypes = ['snare', 'toms', 'hats'];
        const fillType = fillTypes[padNumber - 1];
        console.log(`🥁 Fallback drum mapping: Note ${note} -> Pad ${padNumber} (${fillType})`);
        onDrumPadPress?.(padNumber, velocity / 127);
        return; // Don't process as keyboard note
      }
      
      // Process as keyboard note if not a drum pad
      if (note >= 48 && note <= 96 && velocity > 0) {
        console.log('🎹 Keyboard note pressed:', { note, velocity, noteName: getNoteNameFromMidi(note) });
        onKeyboardNote?.(note, velocity, true); // Note On
      }
      return;
    }
    
    // Handle Note Off messages (status 128 = 0x80)
    if (status === NOTE_OFF_STATUS) {
      const note = dataOrCC;
      
      console.log('🎵 Note Off detected:', { note, noteName: getNoteNameFromMidi(note) });
      
      // Handle keyboard note releases (not drum pads)
      if (note >= 48 && note <= 96 && !DRUM_PAD_NOTES[note]) {
        console.log('🎹 Keyboard note released:', { note, noteName: getNoteNameFromMidi(note) });
        onKeyboardNote?.(note, 0, false); // Note Off
      }
      
      return;
    }
    
    // Handle Pitch Bend (status 224 = 0xE0) for joystick X
    if (status === PITCH_BEND_STATUS) {
      // Pitch bend uses 14-bit resolution: combine dataOrCC (LSB) and value (MSB)
      const pitchBendValue = (value << 7) | dataOrCC;
      const normalizedX = ((pitchBendValue - 8192) / 8192); // Convert 0-16383 to -1 to 1
      
      throttle('joystickX', () => {
        setControllerState(prev => {
          const newState = { ...prev, joystickX: normalizedX };
          onJoystickChange?.(newState.joystickX, newState.joystickY);
          return newState;
        });
      });
      return;
    }
    
    // Handle Control Change messages (status 176 = 0xB0)
    if (status === 176) {
      const cc = dataOrCC;
      const normalizedValue = value / 127; // Convert 0-127 to 0-1
      
      switch (cc) {
        case MIDI_CC.K1_KNOB:
        case MIDI_CC.JOYSTICK_Y:
          // Both K1 knob and joystick Y send CC1 - treat as joystick Y-axis
          throttle('joystickY', () => {
            // Convert 0-1 to 0-1 (bottom to top, only positive values)
            const joystickY = normalizedValue;
            setControllerState(prev => {
              const newState = { ...prev, k1Knob: normalizedValue, joystickY };
              onJoystickChange?.(newState.joystickX, newState.joystickY);
              return newState;
            });
          });
          break;
          
        case MIDI_CC.K2_KNOB:
          throttle('k2', () => {
            setControllerState(prev => ({ ...prev, k2Knob: normalizedValue }));
            onTimbreVolumeChange?.(normalizedValue);
          });
          break;
          
        case MIDI_CC.K3_KNOB:
          throttle('k3', () => {
            setControllerState(prev => ({ ...prev, k3Knob: normalizedValue }));
            onTimbreMixChange?.(normalizedValue);
          });
          break;
          
        case MIDI_CC.K4_KNOB:
          throttle('k4', () => {
            setControllerState(prev => ({ ...prev, k4Knob: normalizedValue }));
            onMasterVolumeChange?.(normalizedValue);
          });
          break;
      }
    }
  };

  const connectMidi = async () => {
    try {
      if (!navigator.requestMIDIAccess) {
        throw new Error('Web MIDI API not supported in this browser');
      }

      const midiAccess = await navigator.requestMIDIAccess();
      
      // Find AKAI MPK Mini or any available input
      let targetInput: WebMidi.MIDIInput | null = null;
      
      for (const input of midiAccess.inputs.values()) {
        if (input.name?.toLowerCase().includes('mpk') || 
            input.name?.toLowerCase().includes('akai')) {
          targetInput = input;
          setDeviceName(input.name);
          break;
        }
      }
      
      // If no AKAI device found, use the first available input
      if (!targetInput && midiAccess.inputs.size > 0) {
        targetInput = midiAccess.inputs.values().next().value;
        setDeviceName(targetInput.name || 'Unknown MIDI Device');
      }
      
      if (targetInput) {
        midiInputRef.current = targetInput;
        targetInput.onmidimessage = handleMidiMessage;
        setIsConnected(true);
        onLog?.(`Connected to MIDI device: ${targetInput.name}`, 'success');
      } else {
        throw new Error('No MIDI input devices found');
      }
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown MIDI error';
      onLog?.(errorMessage, 'error');
      setIsConnected(false);
    }
  };

  const disconnectMidi = () => {
    if (midiInputRef.current) {
      midiInputRef.current.onmidimessage = null;
      midiInputRef.current = null;
    }
    setIsConnected(false);
    setDeviceName(null);
    onLog?.('MIDI device disconnected', 'info');
  };

  useEffect(() => {
    // Auto-connect on mount
    connectMidi();
    
    return () => {
      disconnectMidi();
    };
  }, []);

  return {
    isConnected,
    deviceName,
    controllerState,
    connectMidi,
    disconnectMidi
  };
};