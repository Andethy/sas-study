/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AutomatorAudioProcessorEditor::AutomatorAudioProcessorEditor (AutomatorAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Initialize slider
//    progressSlider.setSliderStyle(juce::Slider::LinearHorizontal);
//    progressSlider.setRange(0.0, 1.0);
//    progressSlider.setValue(0.0);
//    progressSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 100, 20);
//    
//    sliderAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "automation", progressSlider);
//    
//    addAndMakeVisible(progressSlider);
    
    rhythmSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    rhythmSlider.setRange(0.0, 1.0);
    rhythmSlider.setValue(0.0);
    rhythmSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 100, 20);
    
    rhythmAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "rhythm", rhythmSlider);
    addAndMakeVisible(rhythmSlider);
    
    pitchSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    pitchSlider.setRange(0.0, 1.0);
    pitchSlider.setValue(0.0);
    pitchSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 100, 20);
    
    pitchAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "pitch", pitchSlider);
    addAndMakeVisible(pitchSlider);
    
    melodySlider.setSliderStyle(juce::Slider::LinearHorizontal);
    melodySlider.setRange(0.0, 1.0);
    melodySlider.setValue(0.0);
    melodySlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 100, 20);
    
    melodyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "melody", melodySlider);
    addAndMakeVisible(melodySlider);
    
    // For debugging 🏃
    oscLabel.setText("<Waiting for OSC>", juce::dontSendNotification);
    oscLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(oscLabel);
    
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize(400, 300);
}

AutomatorAudioProcessorEditor::~AutomatorAudioProcessorEditor()
{
}

void AutomatorAudioProcessorEditor::updateOsc(const juce::String& msg)
{
    // Update the label with the OSC message
    oscLabel.setText(msg, juce::dontSendNotification);
}

//==============================================================================
void AutomatorAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    g.setColour (juce::Colours::white);
    g.setFont (juce::FontOptions (15.0f));
//    g.drawFittedText ("Hello World!", getLocalBounds(), juce::Justification::centred, 1);
}

void AutomatorAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    rhythmSlider.setBounds(40, 60, getWidth() - 80, 20);
    pitchSlider.setBounds(40, 100, getWidth() - 80, 20);
    melodySlider.setBounds(40, 140, getWidth() - 80, 20);
    
    
    oscLabel.setBounds(40, 180, getWidth() - 80, 20);
    
}
