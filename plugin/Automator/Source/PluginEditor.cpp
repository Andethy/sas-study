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
    progressSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    progressSlider.setRange(0.0, 1.0);
    progressSlider.setValue(0.0);
    progressSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 100, 20);
    
    sliderAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "automation", progressSlider);
    
    addAndMakeVisible(progressSlider);
    
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize(400, 300);
}

AutomatorAudioProcessorEditor::~AutomatorAudioProcessorEditor()
{
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
    progressSlider.setBounds(40, 100, getWidth() - 80, 20);
}
