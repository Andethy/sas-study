#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_osc/juce_osc.h>
#include <juce_audio_utils/juce_audio_utils.h>

// no <vector>

class MOSCAudioProcessor
: public juce::AudioProcessor
, private juce::OSCReceiver
, private juce::OSCReceiver::ListenerWithOSCAddress<juce::OSCReceiver::MessageLoopCallback>
, private juce::Timer
{
public:
    MOSCAudioProcessor();
    ~MOSCAudioProcessor() override;

    //=== AudioProcessor
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override; int getCurrentProgram() override;
    void setCurrentProgram (int) override;
    const juce::String getProgramName (int) override;
    void changeProgramName (int, const juce::String&) override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    bool hasEditor() const override;
    juce::AudioProcessorEditor* createEditor() override;

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // UI helpers
    int  getOscInPort() const;
    void setOscInPort (int newPort);
    bool isOscConnected() const { return oscConnected; }
    int  getMessagesReceived() const { return (int) messagesReceived; }
    juce::String getLastError() const { return lastError; }
    void requestReconnect();

private:
    //=== OSC
    void oscMessageReceived (const juce::OSCMessage& msg) override;     // message thread
    void timerCallback() override;                                      // message thread
    void addAllListeners();
    void removeAllListeners();
    bool connectOnMessageThread (int port);
    void disconnectOnMessageThread();

    //=== Params / state
    juce::AudioProcessorValueTreeState apvts;
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    //=== MIDI handoff
    juce::MidiMessageCollector midiCollector;

    //=== flags / counters
    std::atomic<bool> oscConnected { false };
    std::atomic<int>  messagesReceived { 0 };
    juce::String lastError;

    enum class ReconnectJob { none, reconnectNow } pendingJob { ReconnectJob::none };

    // ---- JUCE-based deferred note-offs (message thread only)
    struct PendingOff { double dueSec; int ch; int note; };
    juce::Array<PendingOff> pendingOffs;  // juce::Array instead of std::vector

    // Small lead so events are never “in the past” for the collector
    static constexpr double kLeadSec   = 0.02;   // 20 ms
    static constexpr double kMinDurSec = 0.005;  // 5 ms

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MOSCAudioProcessor)
};
