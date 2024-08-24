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
class AutomatorAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    AutomatorAudioProcessorEditor (AutomatorAudioProcessor&);
    ~AutomatorAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AutomatorAudioProcessor& audioProcessor;
    
    // Replace with values from the server later
    juce::Slider progressSlider;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> sliderAttachment;
    

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AutomatorAudioProcessorEditor)
};
