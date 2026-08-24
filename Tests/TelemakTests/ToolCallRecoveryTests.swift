import Foundation
import Testing
@testable import Telemak

private let tcOpen = "<tool" + "_call>"
private let tcClose = "</tool" + "_call>"
private let mmOpen = "<minimax:tool" + "_call>"
private let mmClose = "</minimax:tool" + "_call>"
private let fcOpen = "<function" + "_calls>"
private let fcClose = "</function" + "_calls>"

@Test func recoverToolCallsJSONBlock() {
    let text = "Here is the call: \(tcOpen){\"name\": \"web_search\", \"arguments\": {\"query\": \"paris\"}}\(tcClose) and more text."
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].function.name == "web_search")
    #expect(result.content == "Here is the call:  and more text.")
    #expect(result.droppedBlocks == 0)
}

@Test func recoverToolCallsJSONTruncatedString() {
    let text = "First: \(tcOpen){\"name\": \"get_weather\", \"arguments\": {\"city\": \"paris\""
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].function.name == "get_weather")
    #expect(result.droppedBlocks == 0)
}

@Test func recoverToolCallsJSONUnterminatedObject() {
    let text = "\(tcOpen){\"name\": \"book\", \"arguments\": {\"hotel\": \"hilton\""
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].function.name == "book")
    #expect(result.droppedBlocks == 0)
}

@Test func recoverToolCallsJSONWithBrokenEscape() {
    let text = "\(tcOpen){\"name\": \"book\", \"arguments\": {\"note\": \"raw\nnewline\"}}"
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].function.name == "book")
    #expect(result.droppedBlocks == 0)
}

@Test func recoverToolCallsMixedJSONAndXML() {
    let text = "First: \(tcOpen){\"name\": \"a\", \"arguments\": {\"x\": 1}}\(tcClose) then \(mmOpen)<invoke name=\"b\">\n<parameter name=\"q\">paris</parameter>\n\(mmClose) and more text."
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.count == 2)
    #expect(result.toolCalls[0].function.name == "a")
    #expect(result.toolCalls[1].function.name == "b")
    #expect(result.content == "First:  then  and more text.")
    #expect(result.droppedBlocks == 0)
}

@Test func recoverToolCallsDroppedUnrecoverableBlock() {
    let text = "Good: \(tcOpen){\"name\": \"a\", \"arguments\": {}}\(tcClose) and bad: \(mmOpen)not a real call"
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].function.name == "a")
    #expect(result.droppedBlocks == 1)
    #expect(result.content == "Good:  and bad:")
}

@Test func recoverToolCallsNothingRecoverableReturnsOriginalText() {
    let text = "Just plain text with no tool calls at all."
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.isEmpty)
    #expect(result.content == text)
    #expect(result.droppedBlocks == 0)
}

@Test func recoverToolCallsFunctionCallsWrapper() {
    let text = "prefix \(fcOpen)<invoke name=\"search\"><parameter name=\"q\">paris</parameter></invoke>\(fcClose) suffix"
    let result = ChatCompletionsHandler.recoverToolCalls(from: text)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].function.name == "search")
    #expect(result.content == "prefix  suffix")
}
