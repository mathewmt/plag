import React, { useState } from 'react';
import './App.css';
import { TailSpin as Loader } from 'react-loader-spinner';



function App() {
  const [document1, setDocument1] = useState('');
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [similarity, setSimilarity] = useState(null);
  const [loading, setLoading] = useState(false); // Define loading state

  const handleTextChange = (e) => {
    setDocument1(e.target.value);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFileName(file ? file.name : '');
  
    // Read the contents of the file
    const reader = new FileReader();
    reader.onload = async (e) => {
      const text = e.target.result;
      // Now you can send the text content to your backend server
      // using fetch or any other method
      try {
        const response = await fetch('http://localhost:8000/api/v1/preprocess_text_view/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ pdfContent: text }),
        });
  
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
  
        const data = await response.json();
        console.log('Response from server:', data);
      } catch (error) {
        console.error('Error during file upload:', error);
      }
    };
    reader.readAsText(file); // Read file as text
  };
  

  const handlePlagiarismCheck = async () => {
    setLoading(true); // Set loading state to true
    try {
        const response = await fetch('http://localhost:8000/api/v1/preprocess_text_view/', {
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
        setSimilarity(data.similarity);
        console.log('Plagiarism Check Result:', data.result);
    } catch (error) {
        console.error('Error during plagiarism check:', error);
    } finally {
        setLoading(false); // Set loading state back to false
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
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          {loading && <Loader type="TailSpin" color="#00BFFF" height={100} width={100} />}
        </div>

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
        
        {similarity && similarity.length > 0 ? (
          similarity.map((item, index) => (
            <div key={index}>
              <p><a href={item.url} target="_blank" rel="noopener noreferrer">URL: {item.url}</a></p>
              <p>Similarity: {item.similarity.toFixed(1)}</p>
            </div>
          ))
        ) : (
          <p>There is no plagiarism detected.</p>
        )}
      </div>
    </div>
  );
}

export default App;
