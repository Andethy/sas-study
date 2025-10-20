# MOSC - MIDI OSC Plugin

A JUCE-based MIDI effect plugin that receives OSC messages and converts them to MIDI events with precise timing.

## Features

- **OSC to MIDI conversion**: Receives OSC messages and generates timed MIDI events
- **Note impulses**: `/note` messages with configurable duration and velocity
- **Control Change**: `/cc` messages for MIDI CC events
- **Precise timing**: Sample-accurate MIDI scheduling
- **Real-time UI**: Shows connection status, message count, and logging
- **Configurable port**: Change OSC input port from the UI

## OSC Message Formats

### /note - Note Events
```
/note <int: noteNumber> <float: velocity01> [<float: durationSeconds>] [<int: channel>]
```

Examples:
- `/note 60 0.8` - C4 at 80% velocity, 0.1s duration, channel 1
- `/note 64 0.6 0.5` - E4 at 60% velocity, 0.5s duration, channel 1  
- `/note 67 0.9 0.3 2` - G4 at 90% velocity, 0.3s duration, channel 2

### /cc - Control Change
```
/cc <int: ccNumber> <int: value> <int: channel> [<float: offsetMs>]
```

Examples:
- `/cc 1 64 1` - Modulation wheel to 64, channel 1, immediate
- `/cc 7 100 1 100.0` - Volume to 100, channel 1, 100ms delay

## Building

1. Open `Builds/MacOSX/MOSC.xcodeproj` in Xcode
2. Build the AU target: `Product > Build` or `Cmd+B`
3. The plugin will be installed to `~/Library/Audio/Plug-Ins/Components/`

## Testing

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Load the Plugin
- Open your DAW (Logic Pro, Ableton Live, etc.)
- Create a MIDI track
- Load MOSC as a MIDI effect
- The UI should show "Connected" status on port 9001

### 3. Run OSC Tests
```bash
# Test all message types
python test_osc.py

# Test only note messages
python test_osc.py --test note

# Use custom port
python test_osc.py --port 9002
```

### 4. Monitor MIDI Output
- Connect a software instrument after the MOSC plugin
- You should hear notes and see MIDI activity when running the test script
- Check the plugin UI for message count and logging

## UI Features

- **OSC Configuration**: Change input port and reconnect
- **Status Display**: Connection status, message count, error reporting
- **Message Log**: Real-time display of received messages
- **Port Management**: Easy port configuration with validation

## Development Notes

The plugin uses:
- JUCE framework for audio plugin infrastructure
- OSC receiver with sample-accurate MIDI scheduling
- Thread-safe message queuing with spin locks
- Real-time UI updates with timer callbacks

## Troubleshooting

1. **No connection**: Check firewall settings, ensure port isn't in use
2. **No MIDI output**: Verify plugin is loaded as MIDI effect, not audio effect
3. **Timing issues**: Check sample rate, ensure host provides accurate timing info
4. **UI not updating**: Restart plugin or host application

## License

This project is part of the SAS study research.