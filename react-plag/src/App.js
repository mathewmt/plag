import React, { useState } from 'react';
import './App.css';

function App() {
  const [document1, setDocument1] = useState('');
  const [uploadedFileName, setUploadedFileName] = useState('');

  const handleTextChange = (e) => {
    setDocument1(e.target.value);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFileName(file ? file.name : '');
    // You can handle file processing logic here
  };

  const handlePlagiarismCheck = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/plagiarism-check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: document1 }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log('Plagiarism Check Result:', data.result);
    } catch (error) {
      console.error('Error during plagiarism check:', error);
    }
  };

  return (
    <div className="App">
      <h1>Plagiarism Checker</h1>

      <div className="input-container">
        <label htmlFor="document1">Enter Text:</label>
        <textarea
          id="document1"
          placeholder="Enter your text here..."
          value={document1}
          onChange={handleTextChange}
        ></textarea>
      </div>

      <div className="or-separator">
        <span>OR</span>
      </div>

      <div className="input-container">
        <label htmlFor="document2">Upload Document:</label>
        <input
          type="file"
          id="document2"
          onChange={handleFileUpload}
        />
        <div>{uploadedFileName}</div>
      </div>

      <div className="custom-portion">
        <button onClick={handlePlagiarismCheck}>Check Plagiarism</button>
      </div>
    </div>
  );
}

export default App;
