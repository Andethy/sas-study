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
    int port = audioProcessor.parameters.getParameterAsValue("port").getValue();
    DBG("Current value of port " + std::to_string(port));
    
    if (!port) {
        // -> Has to be reference?
        portWindow = std::make_unique<juce::AlertWindow>("Specify Port Number", "Please enter a valid port #:", juce::AlertWindow::QuestionIcon);
        portWindow->addTextEditor("portIn", "", "Port #");
        portWindow->addButton("Submit", 1);
        portWindow->addButton("Cancel", 0);
        portWindow->enterModalState(true, juce::ModalCallbackFunction::create([this](int res)
        {
            DBG("Recieved result");
            if (res == 1) {
                DBG("Result is getPort");
                int portNumber = portWindow->getTextEditor("portIn")->getText().getIntValue();
                DBG("Setting port value");
                audioProcessor.parameters.getParameterAsValue("port").setValue(portNumber);
                DBG("Attempting connection");
                audioProcessor.attemptConnection();
            } else {
                juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Port Required", "The plugin requires a valid port to function.");
            }
            
            portWindow.reset();
        }));
    } else {
        audioProcessor.attemptConnection();
    }
    
    
    
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
    
    portTextEditor.setMultiLine(false);
    portTextEditor.setReturnKeyStartsNewLine(false);
    portTextEditor.setTextToShowWhenEmpty("PORT", juce::Colours::grey);
    portTextEditor.setText(std::to_string((int) audioProcessor.parameters.getParameterAsValue("port").getValue()));
    portTextEditor.addListener(this);
    addAndMakeVisible(portTextEditor);
    
    
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
    portTextEditor.setBounds(180, 220, 50, 20);
    
}

void AutomatorAudioProcessorEditor::textEditorTextChanged (juce::TextEditor& editor)
{
    // When the user types in the TextEditor, update the port number
    if (&editor == &portTextEditor)
    {
        juce::String portText = portTextEditor.getText();
        int newPortNumber = portText.getIntValue();  // Get the number from the text input

        // Validate the port number
        if (newPortNumber > 0 && newPortNumber <= 65535)
        {
            // Update the port number in the AudioProcessorValueTreeState
            audioProcessor.parameters.getParameterAsValue("port").setValue(newPortNumber);

            // Attempt to connect to the new port
            audioProcessor.attemptConnection();
        }
        else
        {
            // Display an error or reset the text if the port number is invalid
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                   "Invalid Port",
                                                   "Please enter a valid port number between 1 and 65535.");
        }
    }
}
