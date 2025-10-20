/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
MOSCAudioProcessorEditor::MOSCAudioProcessorEditor (MOSCAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // OSC Configuration Group
    addAndMakeVisible(oscGroup);
    oscGroup.setText("OSC Configuration");
    
    addAndMakeVisible(portLabel);
    portLabel.setText("Input Port:", juce::dontSendNotification);
    portLabel.attachToComponent(&portEditor, true);
    
    addAndMakeVisible(portEditor);
    portEditor.setText(juce::String(audioProcessor.getOscInPort()));
    portEditor.setInputRestrictions(5, "0123456789");
    
    addAndMakeVisible(connectButton);
    connectButton.setButtonText("Reconnect 🔁");
    connectButton.onClick = [this]() {
        int newPort = portEditor.getText().getIntValue();
        if (newPort > 0 && newPort <= 65535) {
            audioProcessor.setOscInPort(newPort);
        }
    };
    
    // Status Group
    addAndMakeVisible(statusGroup);
    statusGroup.setText("Status");
    
    addAndMakeVisible(connectionStatusLabel);
    addAndMakeVisible(messagesLabel);
    addAndMakeVisible(errorLabel);
    
    // Log Group
    addAndMakeVisible(logGroup);
    logGroup.setText("Message Log");
    
    addAndMakeVisible(logEditor);
    logEditor.setMultiLine(true);
    logEditor.setReadOnly(true);
    logEditor.setScrollbarsShown(true);
    logEditor.setCaretVisible(false);
    
    updateStatus();
    startTimer(100); // Update UI every 100ms
    
    setSize(500, 400);
}

MOSCAudioProcessorEditor::~MOSCAudioProcessorEditor()
{
    stopTimer();
}

//==============================================================================
void MOSCAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
}

void MOSCAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    // OSC Configuration section
    auto oscBounds = bounds.removeFromTop(80);
    oscGroup.setBounds(oscBounds);
    
    auto oscContent = oscBounds.reduced(10, 25);
    auto portRow = oscContent.removeFromTop(25);
    portRow.removeFromLeft(80); // Space for label
    portEditor.setBounds(portRow.removeFromLeft(100));
    portRow.removeFromLeft(10);
    connectButton.setBounds(portRow.removeFromLeft(80));
    
    bounds.removeFromTop(10);
    
    // Status section
    auto statusBounds = bounds.removeFromTop(100);
    statusGroup.setBounds(statusBounds);
    
    auto statusContent = statusBounds.reduced(10, 25);
    connectionStatusLabel.setBounds(statusContent.removeFromTop(25));
    messagesLabel.setBounds(statusContent.removeFromTop(25));
    errorLabel.setBounds(statusContent.removeFromTop(25));
    
    bounds.removeFromTop(10);
    
    // Log section
    logGroup.setBounds(bounds);
    auto logContent = bounds.reduced(10, 25);
    logEditor.setBounds(logContent);
}

void MOSCAudioProcessorEditor::timerCallback()
{
    updateStatus();
    
    // Check for new messages to log
    int currentCount = audioProcessor.getMessagesReceived();
    if (currentCount != lastMessageCount)
    {
        logEditor.insertTextAtCaret("Message " + juce::String(currentCount) + " received\n");
        logEditor.moveCaretToEnd();
        lastMessageCount = currentCount;
    }
}

void MOSCAudioProcessorEditor::updateStatus()
{
    // Connection status
    bool connected = audioProcessor.isOscConnected();
    connectionStatusLabel.setText("Connection: " + juce::String(connected ? "Connected" : "Disconnected"), 
                                  juce::dontSendNotification);
    connectionStatusLabel.setColour(juce::Label::textColourId, 
                                    connected ? juce::Colours::green : juce::Colours::red);
    
    // Message count
    messagesLabel.setText("Messages received: " + juce::String(audioProcessor.getMessagesReceived()), 
                          juce::dontSendNotification);
    
    // Error status
    juce::String error = audioProcessor.getLastError();
    if (error.isNotEmpty())
    {
        errorLabel.setText("Error: " + error, juce::dontSendNotification);
        errorLabel.setColour(juce::Label::textColourId, juce::Colours::red);
    }
    else
    {
        errorLabel.setText("No errors", juce::dontSendNotification);
        errorLabel.setColour(juce::Label::textColourId, juce::Colours::green);
    }
}
