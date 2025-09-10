/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AutomatorAudioProcessor::AutomatorAudioProcessor()
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
    , parameters(*this, nullptr, "PARAMETERS", createParameterLayout()) // CURRENT PORT
{
    isConnected = false;
    int port = parameters.getParameterAsValue("port").getValue();
    if (port) {
        if (!connect(port)) {
            DBG("Failed to connect to port " + std::to_string(port));
        } else {
            DBG("Connected to port " + std::to_string(port));
            OSCReceiver::addListener(this, "/on");
            OSCReceiver::addListener(this, "/rhythm");
            OSCReceiver::addListener(this, "/pitch");
            OSCReceiver::addListener(this, "/melody");
            OSCReceiver::addListener(this, "/note");
            OSCReceiver::addListener(this, "/noteOn");
            OSCReceiver::addListener(this, "/noteOff");
        }
    }
    
}

AutomatorAudioProcessor::~AutomatorAudioProcessor()
{
    disconnect();
}

//==============================================================================

int AutomatorAudioProcessor::attemptConnection()
{
    int port = parameters.getParameterAsValue("port").getValue();  // Retrieve the port number

    if (port == 0)
    {
        DBG("Port number is not set, cannot connect.");
        return 1;  // Do not attempt to connect if port is 0
    }

    if (isConnected)
    {
        disconnect();
        isConnected = false;
    }
    
    DBG("Attempting to connect to port " << port);
    if (!connect(port))
    {
        DBG("Failed to connect to port " << port);
        return 2;
    }
    else
    {
        DBG("Successfully connected to port " << port);
        // ADD LISTENERS 🤦‍♂️
        OSCReceiver::addListener(this, "/on");
        OSCReceiver::addListener(this, "/rhythm");
        OSCReceiver::addListener(this, "/pitch");
        OSCReceiver::addListener(this, "/melody");
        OSCReceiver::addListener(this, "/note");
        OSCReceiver::addListener(this, "/noteOn");
        OSCReceiver::addListener(this, "/noteOff");
        isConnected = true;
    }
    
    return 0;
}

//==============================================================================

void AutomatorAudioProcessor::enqueueImmediate (const juce::MidiMessage& msg, double timestampSeconds)
{
    juce::MidiMessage m = msg;
    m.setTimeStamp(timestampSeconds);
    midiCollector.addMessageToQueue(m);
}

void AutomatorAudioProcessor::enqueueNoteImpulse (int midiNote, float velocity01, double durationSeconds, int channel)
{
    const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
    const uint velocity = (uint) juce::jlimit(0, 127, (int) juce::roundToInt(velocity01 * 127.0f));

    // Note On now
    enqueueImmediate(juce::MidiMessage::noteOn(channel, midiNote, (juce::uint8) velocity), now);

    // Note Off after duration
    const double offTime = now + juce::jmax(0.0, durationSeconds);
    enqueueImmediate(juce::MidiMessage::noteOff(channel, midiNote), offTime);
}

//==============================================================================

juce::AudioProcessorValueTreeState::ParameterLayout AutomatorAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    params.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID("on", 0),
                                                               "On",
                                                               0,
                                                               1,
                                                               1));
    
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("rhythm", 1),
                                                                 "Rhythm",
                                                                 0.0f,
                                                                 1.0f,
                                                                 0.0f
                                                                 ));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("pitch", 2),
                                                                 "Pitch",
                                                                 0.0f,
                                                                 1.0f,
                                                                 0.0f
                                                                 ));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("melody", 3),
                                                                 "Melody",
                                                                 0.0f,
                                                                 1.0f,
                                                                 0.0f
                                                                 ));
    
    
    // DEBUG -> port management
    params.push_back(std::make_unique<juce::AudioParameterInt>(juce::ParameterID("port", 4), "Port", 0, 655535, 0));
//    port = params.at(3).;
    
    return { params.begin(), params.end() };
}

//==============================================================================
//void AutomatorAudioProcessor::run()
//{
//    while (!threadShouldExit())
//    {
//        char message[256] = {};
//        int size = socket.read(message, sizeof(message), false);
//        if (size > 0)
//        {
//            // Message -> float
//            float value = std::stof(message);
//
//            // Bound it to ensure no errors for now
//            value = juce::jlimit(0.0f, 1.0f, value);
//
//            juce::MessageManager::callAsync([this, value]() {
//                parameters.getParameterAsValue("automation").setValue(value);
//            });
//
//            DBG("Received UDP message: " << message);
//        }
//
//        // 😂
//        wait(10);
//    }
//}

void AutomatorAudioProcessor::oscMessageReceived(const juce::OSCMessage& message)
{
    DBG("<TMP> Some message was just recieved.");
    // Verbose OSC diagnostics
    DBG("[OSC] addr=" + message.getAddressPattern().toString() + ", size=" + juce::String(message.size()));
    for (int i = 0; i < message.size(); ++i)
    {
        const auto& arg = message[i];
        juce::String t, v;

        if (arg.isInt32())
        {
            t = "int32";
            v = juce::String(arg.getInt32());
        }
        else if (arg.isFloat32())
        {
            t = "float32";
            v = juce::String(arg.getFloat32());
        }
        else if (arg.isString())
        {
            t = "string";
            v = arg.getString();
        }
        else if (arg.isBlob())
        {
            t = "blob";
            v = "<blob data>";
        }
        else
        {
            t = "unknown";
            v = "<unsupported>";
        }

        DBG("arg[" + juce::String(i) + "] type=" + t + " value=" + v);
    }
    juce::OSCAddressPattern addr = message.getAddressPattern();
    
    // --- MIDI via OSC ---
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
    else if (addr == juce::OSCAddressPattern("/noteOn"))
    {
        // /noteOn <int: noteNumber> <float: velocity01> [<int: channel=1>]
        if (message.size() < 2 || !message[0].isInt32() || !message[1].isFloat32())
        {
            DBG("/noteOn expects: int note, float velocity01, [int channel]");
            return;
        }
        const int note = juce::jlimit(0, 127, message[0].getInt32());
        const float vel01 = juce::jlimit(0.0f, 1.0f, message[1].getFloat32());
        const int channel = (message.size() >= 3 && message[2].isInt32()) ? juce::jlimit(1, 16, message[2].getInt32()) : 1;

        const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
        const uint velocity = (uint) juce::jlimit(0, 127, (int) juce::roundToInt(vel01 * 127.0f));
        enqueueImmediate(juce::MidiMessage::noteOn(channel, note, (juce::uint8) velocity), now);
        return;
    }
    else if (addr == juce::OSCAddressPattern("/noteOff"))
    {
        // /noteOff <int: noteNumber> [<int: channel=1>]
        if (message.size() < 1 || !message[0].isInt32())
        {
            DBG("/noteOff expects: int note, [int channel]");
            return;
        }
        const int note = juce::jlimit(0, 127, message[0].getInt32());
        const int channel = (message.size() >= 2 && message[1].isInt32()) ? juce::jlimit(1, 16, message[1].getInt32()) : 1;

        const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
        enqueueImmediate(juce::MidiMessage::noteOff(channel, note), now);
        return;
    }
    
    // Parameter-style messages expect a single float
    if (addr == juce::OSCAddressPattern("/on") || addr == juce::OSCAddressPattern("/rhythm") ||
        addr == juce::OSCAddressPattern("/pitch") || addr == juce::OSCAddressPattern("/melody"))
    {
        if (message.size() != 1 || !message[0].isFloat32())
        {
            DBG("Parameter messages expect a single float arg.");
            return;
        }
    }
    
    juce::String param;
    
    if (addr == juce::OSCAddressPattern("/on"))
    {
        int value = juce::jlimit(0, 1, static_cast<int>(message[0].getFloat32()));
        DBG("-- on -- " + std::to_string(value) + " -- ");
        juce::MessageManager::callAsync([this, value]() {
            parameters.getParameterAsValue("on").setValue(value);
        });
        return;
    }
        
        
    if (addr == juce::OSCAddressPattern("/rhythm"))
    {
        param = "rhythm";
    }
    else if (addr == juce::OSCAddressPattern("/pitch"))
    {
        param = "pitch";
    }
    else if (addr == juce::OSCAddressPattern("/melody"))
    {
        param = "melody";
    } else {
        DBG("Data sent on wrong address pattern: " + addr.toString());
        return;
    }
    
    float value = (juce::jlimit(0.0f, 1.0f, message[0].getFloat32()));
    
    DBG("-- " + param + " -- " + std::to_string(value) + " -- ");
    std::cout << param << " " << std::to_string(value) << std::endl;

    
//    parameters.getParameter(param)->setValue(juce::jlimit(0.0f, 1.0f, value));
    juce::MessageManager::callAsync([this, param, value]() {
        parameters.getParameterAsValue(param).setValue(value);
    });
//    parameters.getParameterAsValue(param).setValue);
}

//==============================================================================
const juce::String AutomatorAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AutomatorAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AutomatorAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AutomatorAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AutomatorAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AutomatorAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int AutomatorAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AutomatorAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String AutomatorAudioProcessor::getProgramName (int index)
{
    return {};
}

void AutomatorAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void AutomatorAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    midiCollector.reset(sampleRate);

    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    int port = parameters.getParameterAsValue("port").getValue();
    if (port) {
        if (!connect(port)) {
            DBG("Failed to connect to port " + std::to_string(port));
        } else {
            DBG("Connected to port " + std::to_string(port));
            OSCReceiver::addListener(this, "/on");
            OSCReceiver::addListener(this, "/rhythm");
            OSCReceiver::addListener(this, "/pitch");
            OSCReceiver::addListener(this, "/melody");
            OSCReceiver::addListener(this, "/note");
            OSCReceiver::addListener(this, "/noteOn");
            OSCReceiver::addListener(this, "/noteOff");
        }
    }
}

void AutomatorAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool AutomatorAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void AutomatorAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    midiCollector.removeNextBlockOfMessages(midiMessages, buffer.getNumSamples());

    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // In case we have more outputs than inputs, this code clears any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    // This is here to avoid people getting screaming feedback
    // when they first compile a plugin, but obviously you don't need to keep
    // this code if your algorithm always overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // This is the place where you'd normally do the guts of your plugin's
    // audio processing...
    // Make sure to reset the state if your inner loop is processing
    // the samples and the outer loop is handling the channels.
    // Alternatively, you can process the samples with the channels
    // interleaved by keeping the same state.
//    for (int channel = 0; channel < totalNumInputChannels; ++channel)
//    {
//        auto* channelData = buffer.getWritePointer (channel);
//
//        // ..do something to the data...
//        // N/A
//    }
    
    // Should effectively function as mute button
    if (!parameters.getParameterAsValue("on").getValue()) {
        buffer.clear();
    }
}

//==============================================================================
bool AutomatorAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AutomatorAudioProcessor::createEditor()
{
    return new AutomatorAudioProcessorEditor (*this);
}

//==============================================================================
void AutomatorAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::MemoryOutputStream(destData, true).writeString(parameters.state.toXmlString());
}

void AutomatorAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ValueTree tree = juce::ValueTree::fromXml(juce::String::fromUTF8(static_cast<const char*>(data), sizeInBytes));

    if (tree.isValid())
    {
        parameters.state = tree;  // Restore the parameters from the saved state
    }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AutomatorAudioProcessor();
}
