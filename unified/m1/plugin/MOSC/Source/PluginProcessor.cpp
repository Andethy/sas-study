#include "PluginProcessor.h"
#include "PluginEditor.h"

//================ Construction ================
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
, apvts (*this, nullptr, "PARAMS", createLayout())
{
    startTimerHz (250); // small dispatcher for reconnect jobs + MIDI Off notes
}

MOSCAudioProcessor::~MOSCAudioProcessor()
{
    stopTimer();
    disconnectOnMessageThread(); // message thread
}

//================ Introspection ==============
const juce::String MOSCAudioProcessor::getName() const { return JucePlugin_Name; }
bool MOSCAudioProcessor::acceptsMidi() const { return JucePlugin_WantsMidiInput; }
bool MOSCAudioProcessor::producesMidi() const { return JucePlugin_ProducesMidiOutput; }
bool MOSCAudioProcessor::isMidiEffect() const { return JucePlugin_IsMidiEffect; }
double MOSCAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int MOSCAudioProcessor::getNumPrograms() { return 1; }
int MOSCAudioProcessor::getCurrentProgram() { return 0; }
void MOSCAudioProcessor::setCurrentProgram (int) {}
const juce::String MOSCAudioProcessor::getProgramName (int) { return {}; }
void MOSCAudioProcessor::changeProgramName (int, const juce::String&) {}

//================ Lifecycle ==================
void MOSCAudioProcessor::prepareToPlay (double sampleRate, int /*block*/)
{
    midiCollector.reset (sampleRate);

    // (Re)connect using saved port, on message thread
    requestReconnect();
}

void MOSCAudioProcessor::releaseResources()
{
    // Disconnect on the message thread (safe even if not connected)
    disconnectOnMessageThread();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool MOSCAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts); return true;
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

//================ Processing =================
void MOSCAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                       juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals _;

    // Drain any OSC->MIDI we queued with timestamps.
    juce::MidiBuffer fromCollector;
    midiCollector.removeNextBlockOfMessages(fromCollector, buffer.getNumSamples());
    if (fromCollector.getNumEvents() > 0)
        DBG("[AUDIO " << getOscInPort() << "] sending " << fromCollector.getNumEvents() << " MIDI events");
    midi.addEvents(fromCollector, 0, buffer.getNumSamples(), 0);

    buffer.clear(); // MIDI effect
}

//================ Editor =====================
bool MOSCAudioProcessor::hasEditor() const { return true; }
juce::AudioProcessorEditor* MOSCAudioProcessor::createEditor() { return new MOSCAudioProcessorEditor (*this); }

//================ State ======================
juce::AudioProcessorValueTreeState::ParameterLayout MOSCAudioProcessor::createLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;
    p.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID { "port", 1 }, "OSC Input Port", 0, 65535, 9001));
    return { p.begin(), p.end() };
}

void MOSCAudioProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, dest);
}

void MOSCAudioProcessor::setStateInformation (const void* data, int size)
{
    if (auto xml = getXmlFromBinary (data, size))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

//================ Public UI helpers ===========
int MOSCAudioProcessor::getOscInPort() const
{
    return (int) apvts.getParameterAsValue ("port").getValue();
}

void MOSCAudioProcessor::setOscInPort (int newPort)
{
    apvts.getParameterAsValue ("port").setValue (newPort);
    requestReconnect();
}

void MOSCAudioProcessor::requestReconnect()
{
    pendingJob = ReconnectJob::reconnectNow; // timer will perform it on the message thread
}

//================ Timer (message thread) =====
void MOSCAudioProcessor::timerCallback()
{
    if (pendingJob == ReconnectJob::reconnectNow)
    {
        pendingJob = ReconnectJob::none;
        const int port = getOscInPort();
        disconnectOnMessageThread();
        if (port > 0 && connectOnMessageThread (port))
        {
            lastError = {};
            oscConnected = true;
        }
        else
        {
            oscConnected = false;
        }
    }

    // Dispatch due note-offs; remove them using juce::Array::removeIf
    const double nowSec = juce::Time::getMillisecondCounterHiRes() * 0.001;

    pendingOffs.removeIf ([&] (const PendingOff& p)
    {
        if (p.dueSec <= nowSec)
        {
            juce::MidiMessage off = juce::MidiMessage::noteOff (p.ch, p.note);
            off.setTimeStamp (nowSec + kLeadSec);  // tiny lead for collector
            midiCollector.addMessageToQueue (off);
            return true; // remove
        }
        return false; // keep
    });
}

//================ OSC connect/disconnect =====
void MOSCAudioProcessor::disconnectOnMessageThread()
{
    OSCReceiver::removeListener (this);
    removeAllListeners();

    disconnect();
    oscConnected.store (false);

    // Clear any stale scheduled offs + collector
    pendingOffs.clearQuick();
//    midiCollector.discardOutOfDateMessages();
}

bool MOSCAudioProcessor::connectOnMessageThread (int port)
{
    disconnectOnMessageThread();

    if (! connect (port))
    {
        lastError = "OSC: bind failed on port " + juce::String (port);
        DBG (lastError);
        oscConnected.store (false);
        return false;
    }

    addAllListeners();
    lastError = {};
    DBG ("OSC: listening on port " + juce::String (port));
    oscConnected.store (true);

    pendingOffs.clearQuick();
//    midiCollector.discardOutOfDateMessages();
    return true;
}

void MOSCAudioProcessor::addAllListeners()
{
    // Explicit address patterns; message-thread callback class ensures MT safety.
    OSCReceiver::removeListener (this);
    OSCReceiver::addListener (this, juce::OSCAddress{"/note"});
    OSCReceiver::addListener (this, juce::OSCAddress{"/cc"});
}

void MOSCAudioProcessor::removeAllListeners()
{
    OSCReceiver::removeListener (this);
}

//================ OSC callbacks (message thread) ====
void MOSCAudioProcessor::oscMessageReceived (const juce::OSCMessage& message)
{
    messagesReceived.fetch_add (1);
    const auto addr = message.getAddressPattern();
    const double nowSec = juce::Time::getMillisecondCounterHiRes() * 0.001;

    if (addr == juce::OSCAddressPattern ("/note"))
    {
        if (message.size() < 2)
            return;

        const int   note    = juce::jlimit (0, 127, message[0].getInt32());
        const float vel01   = juce::jlimit (0.0f, 1.0f, message[1].getFloat32());
        const double durSec = (message.size() >= 3 && message[2].isFloat32())
                                ? (double) message[2].getFloat32() : 0.1;
        const int ch        = (message.size() >= 4 && message[3].isInt32())
                                ? juce::jlimit (1, 16, message[3].getInt32()) : 1;

        const juce::uint8 velocity = (juce::uint8) juce::jlimit (0, 127, (int) juce::roundToInt (vel01 * 127.0f));

        // ON: small lead so it's never “in the past” for the collector
        juce::MidiMessage on = juce::MidiMessage::noteOn (ch, note, velocity);
        on.setTimeStamp (nowSec + kLeadSec);
        midiCollector.addMessageToQueue (on);

        // OFF: schedule for timer; clamp duration with juce::jmax
        const double durClamped = juce::jmax (kMinDurSec, durSec);
        pendingOffs.add (PendingOff { nowSec + durClamped, ch, note });

        DBG("[OSC " << getOscInPort() << "] queued NOTE ch=" << ch
            << " note=" << note << " vel=" << (int) velocity << " dur=" << durSec);
        return;
    }
    else if (addr == juce::OSCAddressPattern ("/cc"))
    {
        if (message.size() < 3 || !message[0].isInt32() || !message[1].isInt32() || !message[2].isInt32())
            return;

        const int cc  = juce::jlimit (0, 127, message[0].getInt32());
        const int val = juce::jlimit (0, 127, message[1].getInt32());
        const int ch  = juce::jlimit (1, 16, message[2].getInt32());

        juce::MidiMessage m = juce::MidiMessage::controllerEvent (ch, cc, val);
        m.setTimeStamp (nowSec + kLeadSec);
        midiCollector.addMessageToQueue (m);
        return;
    }

    DBG ("Unhandled OSC: " + addr.toString());
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    // Bruh it didn't build until I added this random function
    return new MOSCAudioProcessor();
}
