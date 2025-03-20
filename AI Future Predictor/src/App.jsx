import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FaMicrophone, FaStop } from "react-icons/fa"; // Icons for buttons
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleError = (error, action) => {
    const errorMessage =
      error?.response?.data?.error || `Error ${action} recording`;
    setMessages((prev) => [
      ...prev,
      { role: "system", text: errorMessage, type: "error" },
    ]);
  };

  const startRecording = async () => {
    try {
      setIsRecording(true);
      const response = await axios.get("http://localhost:5001/start-recording");
      setMessages((prev) => [
        ...prev,
        { role: "system", text: response.data.message, type: "info" },
      ]);
    } catch (error) {
      handleError(error, "starting");
      setIsRecording(false);
    }
  };

  const stopRecording = async () => {
    try {
      setIsRecording(false);
      const response = await axios.get("http://localhost:5001/stop-recording");

      setMessages((prev) => [
        ...prev,
        { role: "user", text: response.data.transcription, type: "user" },
        {
          role: "ai",
          text: `Sentiment: ${response.data.sentiment} (Score: ${response.data.sentiment_score})`,
          type: "ai",
        },
        {
          role: "ai",
          text: `Key Points: ${response.data.main_topic}`,
          type: "ai",
        },
        {
          role: "ai",
          text: `Future Prediction: ${response.data.future_prediction}`,
          type: "ai-bold",
        }, // Bold prediction
      ]);
    } catch (error) {
      handleError(error, "stopping");
    }
  };

  return (
    <div className="bg-gray-900 text-white h-screen flex flex-col items-center justify-center">
      <h1 className="text-2xl font-bold mb-4">AI Voice Analysis</h1>

      {/* Chat Window */}
      <div className="w-full max-w-2xl h-96 bg-gray-800 rounded-lg p-4 overflow-y-auto shadow-lg">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`p-3 my-2 max-w-[75%] rounded-xl ${
              msg.role === "user"
                ? "bg-blue-500 ml-auto text-white"
                : msg.type === "ai-bold"
                ? "bg-green-500 text-white font-bold"
                : "bg-gray-700 text-white"
            }`}
          >
            {msg.text}
          </div>
        ))}
        <div ref={chatEndRef}></div>
      </div>

      {/* Buttons */}
      <div className="flex gap-4 mt-4">
        <button
          onClick={startRecording}
          disabled={isRecording}
          className={`flex items-center px-6 py-2 rounded-full text-lg font-semibold transition ${
            isRecording
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-500"
          }`}
        >
          <FaMicrophone className="mr-2" />{" "}
          {isRecording ? "Recording..." : "Start Recording"}
        </button>
        <button
          onClick={stopRecording}
          disabled={!isRecording}
          className={`flex items-center px-6 py-2 rounded-full text-lg font-semibold transition ${
            !isRecording
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-500"
          }`}
        >
          <FaStop className="mr-2" /> Stop Recording
        </button>
      </div>
    </div>
  );
}

export default App;
