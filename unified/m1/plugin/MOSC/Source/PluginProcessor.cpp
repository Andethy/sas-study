/*
  ==============================================================================

    MIDI-effect plugin with OSC receiver/sender and a minimal timed-MIDI queue.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
// Construction

MOSCAudioProcessor::MOSCAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
#endif
{
}

MOSCAudioProcessor::~MOSCAudioProcessor()
{
    disconnectOSC();
}

//==============================================================================
// Introspection

const juce::String MOSCAudioProcessor::getName() const            { return JucePlugin_Name; }
bool MOSCAudioProcessor::acceptsMidi() const                      { return JucePlugin_WantsMidiInput; }
bool MOSCAudioProcessor::producesMidi() const                     { return JucePlugin_ProducesMidiOutput; }
bool MOSCAudioProcessor::isMidiEffect() const                     { return JucePlugin_IsMidiEffect; }
double MOSCAudioProcessor::getTailLengthSeconds() const           { return 0.0; }

int MOSCAudioProcessor::getNumPrograms()                          { return 1; }
int MOSCAudioProcessor::getCurrentProgram()                       { return 0; }
void MOSCAudioProcessor::setCurrentProgram (int)                  {}
const juce::String MOSCAudioProcessor::getProgramName (int)       { return {}; }
void MOSCAudioProcessor::changeProgramName (int, const juce::String&) {}

//==============================================================================
// Lifecycle

void MOSCAudioProcessor::prepareToPlay (double sampleRate, int /*samplesPerBlock*/)
{
    currentSampleRate.store(sampleRate);
    logicalSamplePos.store(0);
    lastHostSamplePos.store(-1);
    currentHostSamplePos.store(0);
    connectOSC();
}

void MOSCAudioProcessor::releaseResources()
{
    disconnectOSC();

    // Clear any pending events
    const juce::SpinLock::ScopedLockType sl (pendingLock);
    pending.clear();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool MOSCAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif
    return true;
  #endif
}
#endif

//==============================================================================
// Processing

void MOSCAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                       juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    
    if (numSamples <= 0)
    {
        // Don’t touch MIDI buffer with offsets; just publish timeline and leave.
        const auto blockStart = logicalSamplePos.load();
        currentHostSamplePos.store(blockStart);
        // do NOT advance logicalSamplePos by 0 (harmless either way)
        buffer.clear();
        return;
    }

    // Start from our always-advancing logical clock.
    juce::int64 blockStart = logicalSamplePos.load();

    if (auto* ph = getPlayHead())
    {
        if (auto posOpt = ph->getPosition()) // std::optional<PositionInfo>
        {
            auto& pos = *posOpt;

            // JUCE Optional types in your build:
            auto timeInSamplesOpt = pos.getTimeInSamples(); // juce::Optional<juce::int64>
            auto isPlayingOpt     = pos.getIsPlaying();     // juce::Optional<bool>

            // Only trust host timeline if both are set and playing.
            if (timeInSamplesOpt.hasValue() && isPlayingOpt)
            {
                const auto hostPos = (juce::int64)*timeInSamplesOpt;
                const auto last    = lastHostSamplePos.load();

                // Only snap when the host is actually advancing.
                if (hostPos != last)
                {
                    blockStart = hostPos;
                    logicalSamplePos.store(hostPos);
                    lastHostSamplePos.store(hostPos);
                }
            }
        }
    }

    // Publish unified timeline and advance it by the block size.
    currentHostSamplePos.store(blockStart);
    logicalSamplePos.store(blockStart + numSamples);

    // Drain everything scheduled before the end of this block (late => offset 0).
    drainEventsIntoBlock(midi, numSamples, blockStart);

    buffer.clear(); // MIDI effect: no audio

}

//==============================================================================
// Editor

bool MOSCAudioProcessor::hasEditor() const                        { return true; }
juce::AudioProcessorEditor* MOSCAudioProcessor::createEditor()    { return new MOSCAudioProcessorEditor (*this); }

//==============================================================================
// State

void MOSCAudioProcessor::getStateInformation (juce::MemoryBlock& destData) {
    juce::MemoryOutputStream mos(destData, true);
    mos.writeInt(oscInPort.load());   // v1: just an int
}
void MOSCAudioProcessor::setStateInformation (const void* data, int sizeInBytes) {
    if (sizeInBytes >= 4)
    {
        juce::MemoryInputStream mis(data, (size_t)sizeInBytes, false);
        const int savedPort = mis.readInt();
        oscInPort.store(savedPort);
        connectOSC();                 // try to connect on project load
    }
}

//==============================================================================
// OSC

void MOSCAudioProcessor::setOscInPort(int port)
{
    if (port != oscInPort.load())
        oscInPort.store(port);
    connectOSC();
}

bool MOSCAudioProcessor::connectOSC()
{
    disconnectOSC();

    const int port = oscInPort.load();
    if (port <= 0 || port > 65535)
    {
        lastError = "Invalid port: " + juce::String(port);
        DBG(lastError);
        oscConnected.store(false);
        return false;
    }

    if (connect(port)) // JUCE OSCReceiver::connect
    {
        OSCReceiver::addListener(this, "/note");
        OSCReceiver::addListener(this, "/cc");
        oscConnected.store(true);
        lastError = {};
        DBG("OSC connected on port " + juce::String(port));
        return true;
    }

    oscConnected.store(false);
    lastError = "Failed to bind port " + juce::String(port) + " (in use?)";
    DBG(lastError);
    return false;
}

void MOSCAudioProcessor::disconnectOSC()
{
    OSCReceiver::removeListener(this);
    disconnect();
    oscConnected.store(false);
}

void MOSCAudioProcessor::oscMessageReceived (const juce::OSCMessage& message)
{
    messagesReceived.fetch_add(1);
    const auto addr = message.getAddressPattern();
    
    if (addr == juce::OSCAddressPattern("/note"))
    {
        // Formats supported:
        //   /note <int: noteNumber> <float: velocity01> <float: durationSeconds> [<int: channel=1>]
        //   /note <int: noteNumber> <float: velocity01>                            (impulse with 0.1s default)
        if (message.size() < 2 || !message[0].isInt32() || !message[1].isFloat32())
        {
            DBG("/note expects: int note, float velocity01, [float durationSec], [int channel]");
            return;
        }

        const int note = juce::jlimit(0, 127, message[0].getInt32());
        const float vel01 = juce::jlimit(0.0f, 1.0f, message[1].getFloat32());
        const double dur = (message.size() >= 3 && message[2].isFloat32()) ? (double) message[2].getFloat32() : 0.1; // default 100ms impulse
        const int channel = (message.size() >= 4 && message[3].isInt32()) ? juce::jlimit(1, 16, message[3].getInt32()) : 1;

        enqueueNoteImpulse(note, vel01, dur, channel);
        return;
    }
    else if (addr == juce::OSCAddressPattern("/cc"))
    {
        // /cc <int: ccNumber> <int: value> <int: channel> [<float: offsetMs>]
        if (message.size() < 3 || !message[0].isInt32() || !message[1].isInt32() || !message[2].isInt32())
        {
            DBG("/cc expects: int ccNumber, int value, int channel, [float offsetMs]");
            return;
        }

        const int cc = juce::jlimit(0, 127, message[0].getInt32());
        const int value = juce::jlimit(0, 127, message[1].getInt32());
        const int channel = juce::jlimit(1, 16, message[2].getInt32());
        const double offsetMs = (message.size() >= 4 && message[3].isFloat32()) ? (double) message[3].getFloat32() : 0.0;

        const auto sr = currentSampleRate.load();
        const auto nowSamples = currentHostSamplePos.load();
        const juce::int64 when = nowSamples + (juce::int64) juce::roundToInt(offsetMs * 0.001 * sr);

        enqueueEvent(juce::MidiMessage::controllerEvent(channel, cc, value), when);
        return;
    }

    // Log unhandled messages
    DBG("Unhandled OSC message: " + addr.toString());
}

//==============================================================================
// Scheduling helpers

void MOSCAudioProcessor::enqueueEvent (const juce::MidiMessage& msg, juce::int64 whenSamples)
{
    const juce::SpinLock::ScopedLockType sl (pendingLock);

    // Keep 'pending' roughly time-sorted (fine for small queues)
    TimedMidiEvent e { msg, whenSamples };
    auto it = std::upper_bound (pending.begin(), pending.end(), e,
                                [] (const TimedMidiEvent& a, const TimedMidiEvent& b)
                                { return a.sampleTime < b.sampleTime; });
    pending.insert (it, std::move (e));
    
    DBG("Enqueued MIDI event: " + msg.getDescription() + " at sample " + juce::String(whenSamples) + " (queue size: " + juce::String(pending.size()) + ")");
}

void MOSCAudioProcessor::enqueueNoteImpulse (int note, float velocity01, double durationSeconds, int channel)
{
    const auto sr = currentSampleRate.load();
    const auto nowSamples = currentHostSamplePos.load();
    const int safety = 32; // a few samples delay to trigger multiple bundled OSC messages simultaneously
    
    const juce::uint8 velocity = (juce::uint8) juce::roundToInt(velocity01 * 127.0f);
    const juce::int64 durationSamples = (juce::int64) juce::roundToInt(durationSeconds * sr);
    
    DBG("Enqueueing note: " + juce::String(note) + " vel:" + juce::String(velocity) + " dur:" + juce::String(durationSeconds) + " ch:" + juce::String(channel));
    DBG("Sample rate: " + juce::String(sr) + ", current sample pos: " + juce::String(nowSamples));
    
    // Schedule note on immediately (use current sample position or slightly ahead)
    const juce::int64 noteOnTime = nowSamples + safety;
    enqueueEvent(juce::MidiMessage::noteOn(channel, note, velocity), noteOnTime);
    
    // Schedule note off after duration
    const juce::int64 noteOffTime = noteOnTime + durationSamples;
    enqueueEvent(juce::MidiMessage::noteOff(channel, note), noteOffTime);
}

void MOSCAudioProcessor::drainEventsIntoBlock (juce::MidiBuffer& midi, int numSamples, juce::int64 blockStartSamples)
{
    if (numSamples <= 0) return;
    
    const juce::int64 blockEnd = blockStartSamples + numSamples;

    std::vector<TimedMidiEvent> due;
    {
        const juce::SpinLock::ScopedLockType sl (pendingLock);

        if (!pending.empty())
        {
            // Take everything scheduled strictly before blockEnd
            auto endIt = std::lower_bound (pending.begin(), pending.end(), blockEnd,
                                           [] (const TimedMidiEvent& e, juce::int64 t)
                                           { return e.sampleTime < t; });

            if (endIt != pending.begin())
            {
                due.assign (pending.begin(), endIt);
                pending.erase (pending.begin(), endIt);
            }
        }
    }

    for (const auto& e : due)
    {
        // Late events go at offset 0; future-in-block get proper offset
        const juce::int64 rawOffset = e.sampleTime - blockStartSamples;
        const int offset = (int) juce::jlimit<juce::int64>(0, numSamples - 1, rawOffset);
        midi.addEvent (e.msg, offset);
        DBG("Added MIDI event at offset " + juce::String(offset) + ": " + e.msg.getDescription());
    }
}

//==============================================================================
// Factory

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MOSCAudioProcessor();
}
