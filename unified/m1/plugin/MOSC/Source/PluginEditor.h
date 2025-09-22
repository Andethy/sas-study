/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class MOSCAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                   private juce::Timer
{
public:
    MOSCAudioProcessorEditor (MOSCAudioProcessor&);
    ~MOSCAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    void updateStatus();
    
    MOSCAudioProcessor& audioProcessor;
    
    // UI Components
    juce::GroupComponent oscGroup;
    juce::Label portLabel;
    juce::TextEditor portEditor;
    juce::TextButton connectButton;
    
    juce::GroupComponent statusGroup;
    juce::Label connectionStatusLabel;
    juce::Label messagesLabel;
    juce::Label errorLabel;
    
    juce::GroupComponent logGroup;
    juce::TextEditor logEditor;
    
    // State
    int lastMessageCount = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MOSCAudioProcessorEditor)
};
