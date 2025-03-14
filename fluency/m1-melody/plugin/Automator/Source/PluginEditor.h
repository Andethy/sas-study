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
class AutomatorAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                       private juce::TextEditor::Listener
{
public:
    AutomatorAudioProcessorEditor (AutomatorAudioProcessor&);
    ~AutomatorAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    
    void textEditorTextChanged (juce::TextEditor&) override;
    
    void updateOsc(const juce::String& msg);

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AutomatorAudioProcessor& audioProcessor;
    
    // Replace with values from the server later
//    juce::Slider progressSlider;
//    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> sliderAttachment;
    
    juce::ToggleButton onButton;
    
    juce::Slider rhythmSlider;
    juce::Slider pitchSlider;
    juce::Slider melodySlider;
    
    juce::TextEditor portTextEditor;
    
    void updatePort();
        
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> onAttachment;
    
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> rhythmAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> pitchAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> melodyAttachment;
    
    std::unique_ptr<juce::AlertWindow> portWindow;
    
    juce::Label oscLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AutomatorAudioProcessorEditor)
};
