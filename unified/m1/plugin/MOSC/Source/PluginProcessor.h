/*
  ==============================================================================

    This file contains the basic framework code for a JUCE MIDI-effect plugin
    with OSC I/O and MIDI scheduling skeleton.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <juce_osc/juce_osc.h>

//==============================================================================
// A simple timed MIDI event container
struct TimedMidiEvent
{
    juce::MidiMessage msg;
    juce::int64 sampleTime = 0; // absolute host sample position
};

//==============================================================================

class MOSCAudioProcessor
    : public juce::AudioProcessor
    , private juce::OSCReceiver
    , private juce::OSCReceiver::ListenerWithOSCAddress<juce::OSCReceiver::MessageLoopCallback>
{
public:
    //==============================================================================
    MOSCAudioProcessor();
    ~MOSCAudioProcessor() override;

    // Audio/MIDI lifecycle
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // Editor
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    // Introspection
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    // Programs
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    // State
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

private:
    //==============================================================================

    // === OSC ===
    void oscMessageReceived (const juce::OSCMessage& message) override;
    bool connectOSC();   // connect receiver/sender based on parameters
    void disconnectOSC();

    // === Scheduling ===
    void enqueueEvent (const juce::MidiMessage& m, juce::int64 whenSamples); // thread-safe
    void enqueueNoteImpulse (int note, float velocity01, double durationSeconds, int channel = 1);
    void drainEventsIntoBlock (juce::MidiBuffer& midi, int numSamples, juce::int64 blockStartSamples);

    // Host timing
    std::atomic<juce::int64> currentHostSamplePos { 0 };
    std::atomic<double> currentSampleRate { 44100.0 };

    // Pending MIDI events, protected by a spin lock (simple & adequate skeleton).
    juce::SpinLock pendingLock;
    std::vector<TimedMidiEvent> pending; // sorted by sampleTime (best-effort)

    // === OSC config ===
    std::atomic<int> oscInPort { 9001 };
    juce::String oscOutHost = "127.0.0.1";
    int oscOutPort = 9002;
    juce::OSCSender oscSender;
    
    // === Status tracking ===
    std::atomic<bool> oscConnected { false };
    std::atomic<int> messagesReceived { 0 };
    juce::String lastError;
    
public:
    // UI accessors
    int getOscInPort() const { return oscInPort.load(); }
    void setOscInPort(int port);
    bool isOscConnected() const { return oscConnected.load(); }
    int getMessagesReceived() const { return messagesReceived.load(); }
    juce::String getLastError() const { return lastError; }

    //==============================================================================

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MOSCAudioProcessor)
};
