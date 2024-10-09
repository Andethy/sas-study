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
    int port = parameters.getParameterAsValue("port").getValue();
    if (port) {
        if (!connect(port)) {
            DBG("Failed to connect to port " + std::to_string(port));
        } else {
            DBG("Connected to port " + std::to_string(port));
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
        return 2;
        DBG("Failed to connect to port " << port);
    }
    else
    {
        DBG("Successfully connected to port " << port);
        // ADD LISTENERS 🤦‍♂️
        OSCReceiver::addListener(this, "/on");
        OSCReceiver::addListener(this, "/rhythm");
        OSCReceiver::addListener(this, "/pitch");
        OSCReceiver::addListener(this, "/melody");
        isConnected = true;
    }
    
    return 0;
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
    
    if (message.size() != 1 || (!message[0].isFloat32())) {
        DBG("Data sent is not in the proper format.");
        return;
    }
    
    juce::OSCAddressPattern addr = message.getAddressPattern();
    
    
    
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
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    int port = parameters.getParameterAsValue("port").getValue();
    if (port) {
        if (!connect(port)) {
            DBG("Failed to connect to port " + std::to_string(port));
        } else {
            DBG("Connected to port " + std::to_string(port));
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
